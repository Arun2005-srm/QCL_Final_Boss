"""Create tiles + CSV while retaining each source scene's group identity."""
import argparse
import csv
from pathlib import Path
from PIL import Image


def prepare(root, output, kind, group_csv=None, independent=False, size=1024):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a fresh preparation output directory")
    if not group_csv and not independent:
        raise ValueError("Supply --groups-csv or explicitly --assume-independent-scenes after overlap audit")
    groups = {}
    if group_csv:
        with open(group_csv, newline="", encoding="utf-8-sig") as f:
            groups = {r["id"]: (r["group"], r["domain"]) for r in csv.DictReader(f)}
    pairs = []
    for image in sorted(root.rglob("*")):
        if image.parent.name != "images" or image.suffix.lower() not in {".tif", ".tiff", ".png", ".jpg"}:
            continue
        mask_dir = image.parent.parent/("labels" if kind == "openearthmap" else "masks")
        masks = [p for p in mask_dir.glob(image.stem+".*") if p.suffix.lower() in {".tif", ".tiff", ".png"}]
        if len(masks) != 1:
            raise ValueError(f"Expected one mask for {image}, found {len(masks)}")
        identifier = image.relative_to(root).with_suffix("").as_posix()
        group, domain = groups.get(identifier, (identifier, image.parent.parent.name if kind == "openearthmap" else "landcover_ai"))
        if group_csv and identifier not in groups:
            raise ValueError(f"Missing source group metadata: {identifier}")
        pairs.append((identifier, image, masks[0], group, domain))
    if not pairs:
        raise ValueError("No source images found; expected images/ and labels/ (OEM) or masks/ (LandCover.ai)")
    (output/"images").mkdir(parents=True)
    (output/"masks").mkdir()
    rows = []
    for index, (identifier, image_path, mask_path, group, domain) in enumerate(pairs):
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            if image.size != mask.size:
                raise ValueError(f"Size mismatch: {identifier}")
            for y in range(0, image.height, size):
                for x in range(0, image.width, size):
                    tile = f"scene_{index:05}_{x}_{y}"
                    box = (x, y, min(x+size, image.width), min(y+size, image.height))
                    image.crop(box).save(output/"images"/(tile+".png"))
                    mask.crop(box).save(output/"masks"/(tile+".png"))
                    rows.append({"id": tile, "image": f"images/{tile}.png", "mask": f"masks/{tile}.png",
                                 "group": group, "domain": domain, "source": identifier, "x": x, "y": y})
    with (output/"samples.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Prepared {len(rows)} tiles from {len(pairs)} source scenes: {output/'samples.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True, choices=["openearthmap", "landcover_ai"])
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--groups-csv")
    parser.add_argument("--assume-independent-scenes", action="store_true")
    parser.add_argument("--tile-size", type=int, default=1024)
    args = parser.parse_args()
    prepare(args.root, args.output, args.kind, args.groups_csv, args.assume_independent_scenes, args.tile_size)
