// SPDX-License-Identifier: Apache-2.0
// bpf-gpu-server: count retired instructions / cycles of a running PID via
// perf_event_open (the system `perf` wrapper is broken on this kernel).
//
//   ./build/perfcount <pid> <ms>
//   -> "instructions=<n> cycles=<n> ms=<n>"
//
// Used to price the bpf-gpu-server path in CPU instructions per generated token:
// count the executor's instructions over a window while it serves N tokens,
// subtract the idle-poll rate measured over an empty window of the same
// length, and divide by the tokens served.
#include <linux/perf_event.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <errno.h>
#include <signal.h>

/* SIGINT stops an open-ended counting window without killing the process
 * before it can print the totals. */
static void on_sigint(int sig) { (void)sig; }

static int open_counter(int pid, unsigned long config)
{
	struct perf_event_attr pe;

	memset(&pe, 0, sizeof(pe));
	pe.type = PERF_TYPE_HARDWARE;
	pe.size = sizeof(pe);
	pe.config = config;
	pe.disabled = 1;

	pe.inherit = 1;
	pe.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED |
			 PERF_FORMAT_TOTAL_TIME_RUNNING;
	return syscall(__NR_perf_event_open, &pe, pid, -1, -1, 0);
}

static long long read_counter(int fd, long long *en, long long *run)
{
	long long v[3] = {0};

	if (read(fd, v, sizeof(v)) != sizeof(v))
		return -1;
	if (en)
		*en = v[1];
	if (run)
		*run = v[2];
	return v[0];
}
int main(int argc, char **argv)
{
	int pid, ms, fdi, fdc;
	long long ins, cyc, en, run;

	if (argc < 3) {
		fprintf(stderr, "usage: %s <pid> <ms>\n", argv[0]);
		return 1;
	}
	pid = atoi(argv[1]);
	ms = atoi(argv[2]);

	fdi = open_counter(pid, PERF_COUNT_HW_INSTRUCTIONS);
	fdc = open_counter(pid, PERF_COUNT_HW_CPU_CYCLES);
	if (fdi < 0 || fdc < 0) {
		perror("perf_event_open");
		return 1;
	}
	ioctl(fdi, PERF_EVENT_IOC_RESET, 0);
	ioctl(fdc, PERF_EVENT_IOC_RESET, 0);
	ioctl(fdi, PERF_EVENT_IOC_ENABLE, 0);
	ioctl(fdc, PERF_EVENT_IOC_ENABLE, 0);
	if (ms > 0) {
		usleep((useconds_t)ms * 1000);
	} else {
		/* ms <= 0: run until SIGINT, so the caller can align the
		 * counting window exactly with a measured load. */
		signal(SIGINT, on_sigint);
		pause();
	}
	ioctl(fdi, PERF_EVENT_IOC_DISABLE, 0);
	ioctl(fdc, PERF_EVENT_IOC_DISABLE, 0);
	ins = read_counter(fdi, &en, &run);
	cyc = read_counter(fdc, NULL, NULL);
	if (ins < 0 || cyc < 0) {
		perror("read");
		return 1;
	}
	printf("instructions=%lld cycles=%lld enabled_ns=%lld ms=%d\n",
	       ins, cyc, en, ms);
	return 0;
}
