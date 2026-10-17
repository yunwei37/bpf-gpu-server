/* SPDX-License-Identifier: Apache-2.0
 * bpf-gpu-server: load and attach the ingress classifier, report its counters.
 *
 *   bpf_gpu_server_load attach [ifname]   create clsact, load, attach tc ingress
 *   bpf_gpu_server_load detach [ifname]   remove the filter and the clsact qdisc
 *   bpf_gpu_server_load tx-load [module.ko] load kernel TX with the pinned map FD
 *   bpf_gpu_server_load stats             print BPF counters and page header
 *
 * The tc ingress hook is used because a `BPF_PROG_TYPE_SOCKET_FILTER` socket
 * filter exposes only helpers, not `skb->remote_ip4`/`remote_port`, on this
 * kernel; tc/clsact is the real ingress hook and is where a NIC's XDP-redirect
 * path would land anyway.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <net/if.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <fcntl.h>

#include <bpf/bpf.h>
#include <bpf/libbpf.h>

#include "bpf_gpu_server_queue.h"

static const char *stats_name[] = {"seen pkts", "ip+udp/ip+tcp pkts",
				   "magic pkts", "drops", "published",
				   "drop: ctl busy", "drop: done busy",
				   "llm magic", "llm busy",
				   "tcp dest", "tcp magic", "stream publish", "sockhash linked", "sockhash failed", "parser calls", "parser invalid"};

int main(int argc, char **argv)
{
	const char *cmd = argc > 1 ? argv[1] : "attach";
	const char *ifname = argc > 2 ? argv[2] : "lo";
	struct bpf_object *bo;
	struct bpf_program *prog;
	int err, ctl_fd, stats_fd;
	struct bgs_page *page;
	struct bpf_tc_hook hook;
	struct bpf_tc_opts opts;
	uint64_t v;
	int i;
	int uring_mode = !strcmp(cmd, "stream-uring-attach");
	int stream_mode = !strcmp(cmd, "stream-attach") || uring_mode;

	if (!strcmp(cmd, "stream-detach")) {
        const char *uring_pins[]={"/sys/fs/bpf/bgs_uring_link",
            "/sys/fs/bpf/bgs_uring_bind_link","/sys/fs/bpf/bgs_uring_owner",
            "/sys/fs/bpf/bgs_pending"};
        for (unsigned j=0;j<sizeof(uring_pins)/sizeof(uring_pins[0]);j++)
            if (unlink(uring_pins[j]) && errno!=ENOENT) { perror("unlink io_uring"); return 1; }
		if (unlink("/sys/fs/bpf/bgs_stream_link") && errno != ENOENT) {
			perror("unlink stream link");
			return 1;
		}
		if (unlink("/sys/fs/bpf/bgs_streams") && errno != ENOENT) {
			perror("unlink stream map"); return 1;
		}
		puts("stream sockops detached");
		return 0;
	}
	if (!strcmp(cmd, "tx-load")) {
		const char *path = argc > 2 ? argv[2] : "module/bpf_gpu_server_tx.ko";
		char params[64];
		int mod_fd = open(path, O_RDONLY | O_CLOEXEC);

		ctl_fd = bpf_obj_get("/sys/fs/bpf/bgs_ctl");
		if (mod_fd < 0 || ctl_fd < 0) {
			perror("open module/map");
			if (mod_fd >= 0) close(mod_fd);
			if (ctl_fd >= 0) close(ctl_fd);
			return 1;
		}
		snprintf(params, sizeof(params), "map_fd=%d", ctl_fd);
		err = syscall(SYS_finit_module, mod_fd, params, 0);
		if (err) perror("finit_module");
		close(mod_fd);
		close(ctl_fd);
		return err ? 1 : 0;
	}
	if (!strcmp(cmd, "detach")) {
		memset(&hook, 0, sizeof(hook));
		hook.sz = sizeof(hook);
		hook.ifindex = if_nametoindex(ifname);
		hook.attach_point = BPF_TC_INGRESS;
		if (!hook.ifindex) {
			fprintf(stderr, "no such interface: %s\n", ifname);
			return 1;
		}
		err = bpf_tc_hook_destroy(&hook);
		if (err && err != -ENOENT && err != -EINVAL) {
			fprintf(stderr, "detach: %s\n", strerror(-err));
			return 1;
		}
		printf("detached from %s\n", ifname);
		return 0;
	}
	if (!strcmp(cmd, "attach") || stream_mode) {
		/* Drop stale pins from a previous run: a fresh object otherwise
		 * cannot take the same pin path, and the daemon would map the
		 * *old* page while the new program writes to the new one. */
		unlink("/sys/fs/bpf/bgs_ctl");
		unlink("/sys/fs/bpf/bgs_stats");
		unlink("/sys/fs/bpf/bgs_db");
        unlink("/sys/fs/bpf/bgs_pending");
	}

	if (!strcmp(cmd, "stats")) {
		/* Read the *pinned* maps: the running program writes to those,
		 * not to a freshly loaded copy. */
		ctl_fd = bpf_obj_get("/sys/fs/bpf/bgs_ctl");
		stats_fd = bpf_obj_get("/sys/fs/bpf/bgs_stats");
		if (ctl_fd < 0 || stats_fd < 0) {
			fprintf(stderr, "no pinned maps: %s\n", strerror(errno));
			return 1;
		}
		page = (struct bgs_page *)mmap(NULL, BGS_PAGE_MMAP_BYTES,
					      PROT_READ | PROT_WRITE,
					      MAP_SHARED, ctl_fd, 0);
		if (page == MAP_FAILED) {
			perror("mmap ctl");
			return 1;
		}
        int context=bpf_obj_get("/sys/fs/bpf/bgs_uring_owner");
        if (context>=0) { stats_name[4]="uring recycled"; close(context); }
		for (i = 0; i < 16; i++) {
			__u32 k = (__u32)i;

			if (bpf_map_lookup_elem(stats_fd, &k, &v) == 0)
				printf("%-14s %llu\n", stats_name[i],
				       (unsigned long long)v);
		}
		printf("head=%u done_seq=%u published=%u acked=%u drops=%u "
		       "ready=%u\n",
		       page->head, page->done_seq, page->published, page->acked,
		       page->drops, page->ready);
		for (i = 0; i < 4 && i < BGS_SLOTS; i++)
			printf("slot[%d]: state=%u id=%u client_ns=%llu x[0]=%f "
			       "| done state=%u id=%u y[0]=%f\n",
			       i, page->slots[i].state, page->slots[i].id,
			       (unsigned long long)page->slots[i].client_ns,
			       page->slots[i].x[0], page->done[i].state,
			       page->done[i].id, page->done[i].y[0]);
		printf("llm_head=%u\n", page->llm_head);
        printf("llm_queued=%u sockhash_first_errno=%u\n",page->llm_queued,page->admission_pad);
        printf("llm_reserved=0x%x\n", page->llm_reserved);
		for (i = 0; i < BGS_LLM_SLOTS; i++)
			printf("llm[%d]: state=%u prompt=%u gen=%u produced=%u client_ns=%llu\n",
			       i, page->llm[i].state, page->llm[i].n_prompt,
			       page->llm[i].n_gen, page->llm[i].produced,
			       (unsigned long long)page->llm[i].client_ns);
		return 0;
	}

	bo = bpf_object__open_file(BGS_BUILD_DIR "/server.bpf.o", NULL);
	if (!bo) {
		perror("open bpf object");
		return 1;
	}
	bpf_object__for_each_program(prog, bo) {
		int stream_program = !strncmp(bpf_program__name(prog), "stream_", 7);
		int uring_program = !strncmp(bpf_program__name(prog), "uring_", 6);
		int verdict = !strcmp(bpf_program__name(prog), "stream_publish");
        int uring_verdict = !strcmp(bpf_program__name(prog), "stream_uring_publish");
        bpf_program__set_autoload(prog, uring_verdict ? uring_mode :
            verdict ? stream_mode && !uring_mode :
            uring_program ? uring_mode : stream_program == stream_mode);
	}
	bpf_map__set_autocreate(bpf_object__find_map_by_name(bo, "streams"), stream_mode);
	bpf_map__set_autocreate(bpf_object__find_map_by_name(bo, "uring_owner"), uring_mode);
    bpf_map__set_autocreate(bpf_object__find_map_by_name(bo,"pending"),uring_mode);
	if (stream_mode) {
		unsigned backlog;
		FILE *setting = fopen("/proc/sys/net/core/somaxconn", "r");
		if (!setting || fscanf(setting, "%u", &backlog) != 1 || !backlog) {
			fprintf(stderr, "cannot discover listener backlog capacity\n");
			return 1;
		}
		fclose(setting);
		bpf_map__set_max_entries(bpf_object__find_map_by_name(bo, "streams"), backlog);
        if(uring_mode)bpf_map__set_max_entries(bpf_object__find_map_by_name(bo,"pending"),backlog);
		printf("sockhash capacity follows somaxconn=%u\n", backlog);
	}

	{
		struct bpf_map *m;

		if (stream_mode) {
			m = bpf_object__find_map_by_name(bo, "streams");
			bpf_map__set_pin_path(m, "/sys/fs/bpf/bgs_streams");
		}
        if (uring_mode) {
            m = bpf_object__find_map_by_name(bo, "uring_owner");
            bpf_map__set_pin_path(m,"/sys/fs/bpf/bgs_uring_owner");
            m=bpf_object__find_map_by_name(bo,"pending");
            bpf_map__set_pin_path(m,"/sys/fs/bpf/bgs_pending");
        }
		m = bpf_object__find_map_by_name(bo, "ctl");
		bpf_map__set_pin_path(m, "/sys/fs/bpf/bgs_ctl");
		m = bpf_object__find_map_by_name(bo, "stats");
		bpf_map__set_pin_path(m, "/sys/fs/bpf/bgs_stats");
		m = bpf_object__find_map_by_name(bo, "doorbell");
		bpf_map__set_pin_path(m, "/sys/fs/bpf/bgs_db");
	}
	if ((err = bpf_object__load(bo))) {
		fprintf(stderr, "load: %s\n", strerror(-err));
		return 1;
	}
	ctl_fd = bpf_map__fd(bpf_object__find_map_by_name(bo, "ctl"));
	stats_fd = bpf_map__fd(bpf_object__find_map_by_name(bo, "stats"));
	page = (struct bgs_page *)mmap(NULL, BGS_PAGE_MMAP_BYTES, PROT_READ | PROT_WRITE,
				      MAP_SHARED, ctl_fd, 0);
	if (page == MAP_FAILED) {
		perror("mmap ctl");
		return 1;
	}

	if (stream_mode) {
		char line[4096], path[8192];
		FILE *cg = fopen("/proc/self/cgroup", "r");
		struct bpf_link *link;
		int sockmap = bpf_map__fd(bpf_object__find_map_by_name(bo, "streams"));
		int cgroup;

		if (!cg || !fgets(line, sizeof(line), cg) || strncmp(line, "0::", 3)) {
			fprintf(stderr, "cannot discover current cgroup v2\n");
			return 1;
		}
		fclose(cg);
		line[strcspn(line, "\n")] = 0;
		snprintf(path, sizeof(path), "/sys/fs/cgroup%s", line + 3);
		cgroup = open(path, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
		if (cgroup < 0) { perror("current cgroup"); return 1; }
		prog = bpf_object__find_program_by_name(bo, "stream_parse");
		if (bpf_prog_attach(bpf_program__fd(prog), sockmap, BPF_SK_SKB_STREAM_PARSER, 0)) {
			perror("stream parser attach"); return 1;
		}
		prog = bpf_object__find_program_by_name(bo, uring_mode ? "stream_uring_publish" : "stream_publish");
		if (bpf_prog_attach(bpf_program__fd(prog), sockmap, BPF_SK_SKB_STREAM_VERDICT, 0)) {
			perror("stream verdict attach"); return 1;
		}
		prog = bpf_object__find_program_by_name(bo, "stream_sockops");
		link = bpf_program__attach_cgroup(prog, cgroup);
		if (!link || libbpf_get_error(link)) { fprintf(stderr, "sockops attach failed\n"); return 1; }
		if (bpf_link__pin(link, "/sys/fs/bpf/bgs_stream_link")) {
			bpf_link__destroy(link); perror("stream link pin"); return 1;
		}
        if (uring_mode) {
            const char *names[]={"uring_bind","uring_recycle"};
            const char *events[]={"io_uring_create","io_uring_complete"};
            const char *paths[]={"/sys/fs/bpf/bgs_uring_bind_link","/sys/fs/bpf/bgs_uring_link"};
            for (unsigned j=0;j<2;j++) {
                prog=bpf_object__find_program_by_name(bo,names[j]);
                link=bpf_program__attach_tracepoint(prog,"io_uring",events[j]);
                if (!link || libbpf_get_error(link)) { fprintf(stderr,"io_uring attach failed\n"); return 1; }
                if (bpf_link__pin(link,paths[j])) { bpf_link__destroy(link);perror("io_uring pin");return 1; }
            }
        }
		close(cgroup);
		puts("stream parser/verdict and sockops attached to current Workspace cgroup");
		return 0;
	}

	memset(&hook, 0, sizeof(hook));
	hook.sz = sizeof(hook);
	hook.ifindex = if_nametoindex(ifname);
	hook.attach_point = BPF_TC_INGRESS;
	if (hook.ifindex == 0) {
		fprintf(stderr, "no such interface: %s\n", ifname);
		return 1;
	}


	/* clsact survives process exits, and every attach adds another
	 * filter, so old ones keep running (and keep bumping the counters)
	 * after new ones are added. Start from a clean hook. */
	bpf_tc_hook_destroy(&hook);
	if (bpf_tc_hook_create(&hook) && errno != EEXIST) {
		fprintf(stderr, "hook create: %s\n", strerror(errno));
		return 1;
	}
	prog = bpf_object__find_program_by_name(bo, "ingress");
	memset(&opts, 0, sizeof(opts));
	opts.sz = sizeof(opts);
	opts.prog_fd = bpf_program__fd(prog);
	if ((err = bpf_tc_attach(&hook, &opts))) {
		fprintf(stderr, "tc attach: %s\n", strerror(-err));
		return 1;
	}
	{
		struct bpf_map_info info = {};
		__u32 len = sizeof(info);

		if (bpf_obj_get_info_by_fd(ctl_fd, &info, &len)) {
			perror("map info");
			return 1;
		}
		printf("attached to %s (prog_id %u), ctl map id=%u size=%u page=%p\n",
		       ifname, opts.prog_id, info.id, info.value_size, (void *)page);
	}
	return 0;
}
