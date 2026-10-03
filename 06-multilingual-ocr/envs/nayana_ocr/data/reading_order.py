"""Deterministic geometric reading order, not a claim of canonical semantic order.

Separate spanning blocks, then split at vertical whitespace (columns), then
horizontal whitespace (bands). Recurse into each partition. If no whitespace cut
exists, use top edge then reading-direction x edge and region ID as tie breakers.
No language-model reconstruction or source region-list ordering is used.
"""


def ordered_regions(regions, *, rtl=False):
    if len(regions) < 2:
        return list(regions)
    span = max(r["bbox"][2] for r in regions) - min(r["bbox"][0] for r in regions)
    # Peel off spanning headings/footers before finding column gutters. Otherwise
    # a footer connects both columns and a horizontal cut can interleave them.
    for region in sorted(regions, key=lambda r: (r["bbox"][1], str(r["region_id"]))):
        box = region["bbox"]
        if box[2] - box[0] < 0.8 * span:
            continue
        others = [r for r in regions if r is not region]
        above = [r for r in others if r["bbox"][3] <= box[1]]
        below = [r for r in others if r["bbox"][1] >= box[3]]
        if len(above) + len(below) == len(others):
            return (
                ordered_regions(above, rtl=rtl)
                + [region]
                + ordered_regions(below, rtl=rtl)
            )
    for axis in (0, 1):
        intervals = sorted((r["bbox"][axis], r["bbox"][axis + 2]) for r in regions)
        end, gaps = intervals[0][1], []
        for start, stop in intervals[1:]:
            if start > end:
                gaps.append((start - end, (start + end) / 2))
            end = max(end, stop)
        if not gaps:
            continue
        # Largest whitespace cut; ties prefer the first cut in geometric order.
        _, cut = max(gaps, key=lambda gap: (gap[0], -gap[1]))
        before = [r for r in regions if r["bbox"][axis + 2] <= cut]
        after = [r for r in regions if r["bbox"][axis] >= cut]
        partitions = (after, before) if axis == 0 and rtl else (before, after)
        return [r for group in partitions for r in ordered_regions(group, rtl=rtl)]
    return sorted(
        regions,
        key=lambda r: (
            r["bbox"][1],
            -r["bbox"][2] if rtl else r["bbox"][0],
            str(r["region_id"]),
        ),
    )


def overlapping_regions(regions):
    """Overlapping annotation boxes make concatenation potentially duplicate text."""
    for index, first in enumerate(regions):
        a = first["bbox"]
        for second in regions[index + 1 :]:
            b = second["bbox"]
            if min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1]):
                return True
    return False
