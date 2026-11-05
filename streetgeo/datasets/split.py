"""Freeze train/validation/test splits by sequence and geographic block."""

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os

from shapely.geometry import Point, box
from shapely.strtree import STRtree


def distance_m(left, right):
    lat1, lon1 = map(math.radians, (left["latitude"], left["longitude"]))
    lat2, lon2 = map(math.radians, (right["latitude"], right["longitude"]))
    a = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 12742017.6 * math.asin(math.sqrt(min(1, max(0, a))))


def remove_nearby(candidates, reference, guard_m, cancelled=None):
    if guard_m == 0 or not reference:
        return candidates
    points = [Point(row["longitude"], row["latitude"]) for row in reference]
    tree = STRtree(points)
    kept = []
    for index, row in enumerate(candidates):
        if cancelled is not None and index % 512 == 0:
            from streetgeo.datasets.sources import check_cancel
            check_cancel(cancelled)
        lon, lat = row["longitude"], row["latitude"]
        # Conservative bounds, followed by exact spherical distance checks.
        dlat = guard_m / 110000
        dlon = min(360, dlat / max(1e-8, math.cos(math.radians(min(89.999999, abs(lat) + dlat)))))
        queries = [box(lon - dlon, lat - dlat, lon + dlon, lat + dlat)]
        if lon - dlon < -180:
            queries.append(box(lon - dlon + 360, lat - dlat, 180, lat + dlat))
        if lon + dlon > 180:
            queries.append(box(-180, lat - dlat, lon + dlon - 360, lat + dlat))
        indices = {int(index) for query in queries for index in tree.query(query)}
        if not any(distance_m(row, reference[index]) < guard_m for index in indices):
            kept.append(row)
    return kept


def create_splits(rows, *, seed=1337, block_degrees=0.25, guard_m=250, ratios=(0.8, 0.1, 0.1), cancelled=None,
                  stratify_cities=False):
    if (not math.isfinite(block_degrees) or not math.isfinite(guard_m) or
            block_degrees <= 0 or guard_m < 0 or any(value <= 0 for value in ratios) or not math.isclose(sum(ratios), 1)):
        raise ValueError("positive block size/ratios, nonnegative guard, ratios summing to one required")
    parent = list(range(len(rows)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        parent[find(left)] = find(right)

    sequences, blocks, identities = {}, {}, {}
    for index, row in enumerate(rows):
        if cancelled is not None and index % 512 == 0:
            from streetgeo.datasets.sources import check_cancel
            check_cancel(cancelled)
        sequence = row.get("sequence_id")
        if sequence:
            identity = (str(row.get("source", "")).lower(), str(sequence))
            if identity in sequences:
                union(index, sequences[identity])
            sequences[identity] = index
        source = str(row.get("source", "")).lower()
        keys = []
        for name in ("group_id", "panorama_id"):
            if row.get(name):
                keys.append((name, source, str(row[name])))
        if row.get("place_id") is not None:
            keys.append(("place", source, str(row.get("city", "")), str(row["place_id"])))
        # Repeated references to the same file must never cross partitions.
        for path in row.get("images", [row["image"]] if "image" in row else []):
            keys.append(("image", os.path.normcase(str(path))))
        for key in keys:
            if key in identities:
                union(index, identities[key])
            identities[key] = index
    for index, row in enumerate(rows):
        latitude, longitude = row["latitude"], row["longitude"]
        block = (math.floor(longitude / block_degrees), math.floor(latitude / block_degrees))
        if block in blocks:
            union(index, blocks[block])
        blocks[block] = index
    groups = defaultdict(list)
    for index in range(len(rows)):
        groups[find(index)].append(index)
    if len(groups) < 3:
        raise ValueError("Need at least three independent sequence/geographic groups; collect more locations")

    def order(group):
        identity = ",".join(str(rows[index].get("mapillary_id", rows[index].get("image", index))) for index in group)
        return -len(group), hashlib.sha256(f"{seed}:{identity}".encode()).hexdigest()

    ordered = sorted(groups.values(), key=order)
    partitions = [[], [], []]
    city_totals, assigned_cities = defaultdict(int), [defaultdict(int) for _ in range(3)]
    if stratify_cities:
        for row in rows:
            city_totals[(row.get("source", "custom"), row.get("city", "unspecified"))] += 1
    for index, group in enumerate(ordered):
        empty = [number for number in range(3) if not partitions[number]]
        choices = empty if len(ordered) - index <= len(empty) else range(3)
        if stratify_cities:
            group_cities = defaultdict(int)
            for item in group:
                row = rows[item]
                group_cities[(row.get("source", "custom"), row.get("city", "unspecified"))] += 1
            def deficit(number):
                return sum(count * (ratios[number] - assigned_cities[number][city] / city_totals[city])
                           for city, count in group_cities.items()) / len(group)
            chosen = max(choices, key=lambda number: (deficit(number), len(rows) * ratios[number] - len(partitions[number])))
            for city, count in group_cities.items():
                assigned_cities[chosen][city] += count
        else:
            chosen = max(choices, key=lambda number: len(rows) * ratios[number] - len(partitions[number]))
        partitions[chosen].extend(rows[item] for item in group)
    original_counts = [len(part) for part in partitions]
    train, val, test = partitions
    val = remove_nearby(val, train, guard_m, cancelled)
    test = remove_nearby(test, train + val, guard_m, cancelled)
    if not train or not val or not test:
        raise ValueError("A holdout split is empty after geographic buffering; collect more independent locations")
    return {"train": train, "val": val, "test": test}, {
        "groups": len(groups), "largest_group": max(map(len, groups.values())),
        "before_buffer": dict(zip(("train", "val", "test"), original_counts)),
        "counts": {"train": len(train), "val": len(val), "test": len(test)},
        "dropped_near_holdout_boundary": sum(original_counts) - len(train) - len(val) - len(test),
        "seed": seed, "block_degrees": block_degrees, "guard_m": guard_m,
    }


def prepare(manifest, *, output_dir=None, seed=1337, block_degrees=0.25, guard_m=250):
    from streetgeo.datasets.prepare import prepare_manifest

    report = prepare_manifest(manifest, output_dir=output_dir, seed=seed,
                              block_degrees=block_degrees, guard_m=guard_m)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/mapillary/all.jsonl")
    parser.add_argument("--output-dir", help="frozen split directory; default <manifest directory>/splits")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--block-degrees", type=float, default=0.25)
    parser.add_argument("--guard-m", type=float, default=250)
    args = parser.parse_args()
    prepare(args.manifest, output_dir=args.output_dir, seed=args.seed,
            block_degrees=args.block_degrees, guard_m=args.guard_m)


if __name__ == "__main__":
    main()
