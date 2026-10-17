// SPDX-License-Identifier: Apache-2.0
// bpf-gpu-server: minimal ingress probe.
//
// Attached as a classic-style socket filter (SO_ATTACH_BPF) to a UDP socket.
// For every packet it records
//   (payload[0..8) as sent-timestamp, bpf_ktime_get_ns())
// into a ring buffer. The purpose is to measure, on this kernel, the cost of
// the only wakeup primitive a *programmable* eBPF path can offer a userspace
// GPU executor: a pollable BPF map (ring buffer) that wakes a userspace thread.
//
// Carrying the sender timestamp inside the record makes userspace pairing
// exact even if packets are lost or reordered.
#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>

struct evt {
	__u64 send_ts; /* CLOCK_MONOTONIC taken by the sender in the payload */
	__u64 t_ns;    /* bpf_ktime_get_ns() at socket-filter time */
	__u32 len;     /* payload length seen */
	__u32 pad;
};

struct {
	__uint(type, BPF_MAP_TYPE_RINGBUF);
	__uint(max_entries, 1 << 20);
} events SEC(".maps");

SEC("socket")
int ingress(struct __sk_buff *skb)
{
	struct evt *e;
	__u64 send_ts = 0;

	e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
	/* For a UDP socket filter, offset 0 is the UDP header (verified on
	 * this kernel), so the payload starts at offset 8. */
	long r = bpf_skb_load_bytes(skb, 8, &send_ts, sizeof(send_ts));

	if (e) {
		e->send_ts = send_ts;
		e->t_ns = bpf_ktime_get_ns();
		e->len = skb->len;
		e->pad = (int)r;
		bpf_ringbuf_submit(e, 0);
	}
	/* Accept everything; this program only observes. */
	return 0xffffffff;
}

char _license[] SEC("license") = "GPL";
