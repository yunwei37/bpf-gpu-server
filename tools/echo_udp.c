/* needed for CPU_ZERO/CPU_SET and sched_setaffinity before any header */
#define _GNU_SOURCE
/* SPDX-License-Identifier: Apache-2.0
 * Baseline: plain UDP userspace echo server, one thread, pinned.
 *
 * This is the "ordinary userspace round trip" the bpf-gpu-server path is compared
 * against: recvfrom -> (optional work) -> sendto, no eBPF, no control page, no
 * GPU. It replies with the first 16 floats of the request so the wire cost is
 * comparable.
 */
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

struct req {
	uint32_t magic;
	uint32_t pad;
	uint64_t client_ns;
	float x[16];
};

struct resp {
	uint32_t magic;
	uint32_t id;
	uint32_t pad;
	uint32_t pad2;
	uint64_t client_ns;
	uint64_t ingress_ns;
	uint64_t gpu_start_ns;
	uint64_t gpu_done_ns;
	float y[16];
};

int main(int argc, char **argv)
{
	int port = argc > 1 ? atoi(argv[1]) : 39401;
	int cpu = argc > 2 ? atoi(argv[2]) : 4;
	int seconds = argc > 3 ? atoi(argv[3]) : 5;
	struct sockaddr_in addr;
	int s;

	if (cpu >= 0) {
		cpu_set_t set;

		CPU_ZERO(&set);
		CPU_SET(cpu, &set);
		sched_setaffinity(0, sizeof(set), &set);
	}
	s = socket(AF_INET, SOCK_DGRAM, 0);
	memset(&addr, 0, sizeof(addr));
	addr.sin_family = AF_INET;
	addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
	addr.sin_port = htons((uint16_t)port);
	if (bind(s, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
		perror("bind");
		return 1;
	}
	{
		int sz = 8 << 20;

		setsockopt(s, SOL_SOCKET, SO_RCVBUF, &sz, sizeof(sz));
	}
	printf("echo_udp: port=%d cpu=%d\n", port, cpu);
	fflush(stdout);
	while (1) {
		struct req r;
		struct resp out;
		struct sockaddr_in from;
		socklen_t fl = sizeof(from);
		ssize_t n = recvfrom(s, &r, sizeof(r), 0,
				     (struct sockaddr *)&from, &fl);

		if (n <= 0)
			break;
		if (r.magic != 0x46504251u)
			continue;
		memset(&out, 0, sizeof(out));
		out.magic = r.magic;
		out.id = 0;
		out.client_ns = r.client_ns;
		out.ingress_ns = 0;
		out.gpu_start_ns = 0;
		out.gpu_done_ns = 0;
		memcpy(out.y, r.x, sizeof(out.y));
		sendto(s, &out, sizeof(out), 0, (struct sockaddr *)&from, fl);
		(void)seconds;
	}
	return 0;
}
