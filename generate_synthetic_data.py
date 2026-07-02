import json
import random
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from pathlib import Path

@dataclass
class CADSample:
    text_input: str
    cad_output: Dict[str, Any]
    metadata: Dict[str, Any]

class SyntheticDataGenerator:
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.templates = self._build_templates()

    def _build_templates(self) -> List[Dict]:
        return [
            {
                "name": "box_with_hole",
                "text_templates": [
                    "Create a {L}mm x {W}mm x {H}mm rectangular block with a {D}mm diameter through-hole in the center",
                    "Make a {L} by {W} by {H} box with a centered {D}mm hole through it",
                    "Design a {L}x{W}x{H}mm block featuring a {D}mm central bore",
                ],
                "params": {
                    "L": (20, 100), "W": (15, 80), "H": (10, 80), "D": (3, 30)
                },
                "cad_template": {
                    "operations": [
                        {"type": "box", "params": {"length": "L", "width": "W", "height": "H"}},
                        {"type": "hole", "params": {"diameter": "D"}}
                    ]
                }
            },
            {
                "name": "cylinder_with_keyway",
                "text_templates": [
                    "Design a cylindrical shaft {L}mm long with {D}mm diameter, with a {K}mm keyway slot along its length",
                    "Create a {L}mm long cylinder of {D}mm diameter featuring a {K}mm wide keyway",
                    "Make a shaft {L}x{D}mm with {K}mm keyway running the full length",
                ],
                "params": {"L": (50, 200), "D": (10, 50), "K": (3, 12)},
                "cad_template": {
                    "operations": [
                        {"type": "cylinder", "params": {"radius": "D/2", "height": "L"}},
                        {"type": "box", "params": {"length": "L", "width": "K", "height": "K"}}
                    ]
                }
            },
            {
                "name": "box_with_fillets",
                "text_templates": [
                    "Make a {S}mm cube with {R}mm fillets on all edges",
                    "Create a {S}x{S}x{S}mm cube with {R}mm rounded edges",
                    "Design a {S}mm cube featuring {R}mm fillets on every edge",
                ],
                "params": {"S": (20, 80), "R": (1, 10)},
                "cad_template": {
                    "operations": [
                        {"type": "box", "params": {"length": "S", "width": "S", "height": "S"}},
                        {"type": "fillet", "params": {"radius": "R", "edges": "|Z"}}
                    ]
                }
            },
            {
                "name": "hollow_cylinder",
                "text_templates": [
                    "Create a hollow cylinder with {OD}mm outer diameter, {ID}mm inner diameter, and {H}mm height",
                    "Make a tube {OD}mm OD, {ID}mm ID, {H}mm tall",
                    "Design a cylindrical shell: outer {OD}mm, inner {ID}mm, height {H}mm",
                ],
                "params": {"OD": (20, 80), "ID": (10, 60), "H": (20, 100)},
                "cad_template": {
                    "operations": [
                        {"type": "cylinder", "params": {"radius": "OD/2", "height": "H"}},
                        {"type": "cylinder", "params": {"radius": "ID/2", "height": "H"}}
                    ]
                }
            },
            {
                "name": "bracket_with_holes",
                "text_templates": [
                    "Design a bracket: {L}mm x {W}mm base plate, {T}mm thick, with two {D}mm mounting holes {S}mm apart, and a {H}mm tall vertical flange with {R}mm fillets",
                    "Create an L-bracket {L}x{W}x{T}mm with two {D}mm holes spaced {S}mm, vertical flange {H}mm with {R}mm fillets",
                    "Make a mounting bracket: base {L}x{W}x{T}, two {D}mm holes {S}mm centers, flange {H}mm tall, {R}mm fillets",
                ],
                "params": {"L": (40, 100), "W": (30, 80), "T": (3, 10), "D": (5, 12), "S": (20, 70), "H": (20, 60), "R": (1, 5)},
                "cad_template": {
                    "operations": [
                        {"type": "box", "params": {"length": "L", "width": "W", "height": "T"}},
                        {"type": "hole", "params": {"diameter": "D"}},
                        {"type": "box", "params": {"length": "L", "width": "10", "height": "H"}},
                        {"type": "fillet", "params": {"radius": "R", "edges": "|Z"}}
                    ]
                }
            },
            {
                "name": "conical_frustum",
                "text_templates": [
                    "Create a conical frustum: bottom radius {R1}mm, top radius {R2}mm, height {H}mm",
                    "Make a truncated cone with base radius {R1}, top radius {R2}, height {H}",
                    "Design a frustum: bottom {R1}mm, top {R2}mm, {H}mm tall",
                ],
                "params": {"R1": (15, 40), "R2": (5, 25), "H": (20, 60)},
                "cad_template": {
                    "operations": [
                        {"type": "cylinder", "params": {"radius": "R1", "height": "H"}},
                        {"type": "cylinder", "params": {"radius": "R2", "height": "H"}}
                    ]
                }
            },
            {
                "name": "box_with_counterbore",
                "text_templates": [
                    "Create a {L}x{W}x{H}mm block with a {D1}mm counterbore {D2}mm deep for a socket head cap screw",
                    "Make a {L} by {W} by {H} block featuring a {D1}mm x {D2}mm counterbored hole",
                    "Design a {L}x{W}x{H}mm plate with {D1}mm counterbore {D2}mm deep",
                ],
                "params": {"L": (30, 80), "W": (30, 80), "H": (10, 30), "D1": (8, 20), "D2": (5, 15)},
                "cad_template": {
                    "operations": [
                        {"type": "box", "params": {"length": "L", "width": "W", "height": "H"}},
                        {"type": "hole", "params": {"diameter": "D1"}},
                        {"type": "cylinder", "params": {"radius": "D1/2", "height": "D2"}}
                    ]
                }
            },
            {
                "name": "flanged_cylinder",
                "text_templates": [
                    "Design a flanged cylinder: shaft {D1}mm diameter x {L1}mm long, flange {D2}mm diameter x {T}mm thick",
                    "Create a cylinder {D1}x{L1}mm with a {D2}mm diameter flange {T}mm thick at the base",
                    "Make a stepped shaft: {D1}mm x {L1}mm body, {D2}mm x {T}mm flange",
                ],
                "params": {"D1": (10, 40), "L1": (30, 120), "D2": (30, 80), "T": (5, 20)},
                "cad_template": {
                    "operations": [
                        {"type": "cylinder", "params": {"radius": "D1/2", "height": "L1"}},
                        {"type": "cylinder", "params": {"radius": "D2/2", "height": "T"}}
                    ]
                }
            },
            {
                "name": "box_with_pocket",
                "text_templates": [
                    "Create a {L}x{W}x{H}mm block with a rectangular pocket {PL}x{PW}x{PH}mm centered on top face",
                    "Make a {L} by {W} by {H} block with a {PL}x{PW}x{PH}mm cavity",
                    "Design a {L}x{W}x{H}mm plate featuring a centered pocket {PL}x{PW}x{PH}mm",
                ],
                "params": {"L": (40, 100), "W": (40, 100), "H": (15, 40), "PL": (10, 50), "PW": (10, 50), "PH": (5, 20)},
                "cad_template": {
                    "operations": [
                        {"type": "box", "params": {"length": "L", "width": "W", "height": "H"}},
                        {"type": "box", "params": {"length": "PL", "width": "PW", "height": "PH"}}
                    ]
                }
            },
            {
                "name": "multi_hole_pattern",
                "text_templates": [
                    "Create a {L}x{W}x{T}mm plate with a {N}x{M} grid of {D}mm holes spaced {S}mm apart",
                    "Make a {L} by {W} by {T} plate with {N} rows and {M} columns of {D}mm holes on {S}mm centers",
                    "Design a {L}x{W}x{T}mm mounting plate with {N}x{M} hole pattern, {D}mm diameter, {S}mm spacing",
                ],
                "params": {"L": (60, 150), "W": (60, 150), "T": (5, 15), "N": (2, 5), "M": (2, 5), "D": (4, 10), "S": (15, 40)},
                "cad_template": {
                    "operations": [
                        {"type": "box", "params": {"length": "L", "width": "W", "height": "T"}},
                        {"type": "hole", "params": {"diameter": "D"}}
                    ]
                }
            },
            {
                "name": "stepped_shaft",
                "text_templates": [
                    "Design a stepped shaft: {D1}mm diameter x {L1}mm, {D2}mm diameter x {L2}mm, {D3}mm diameter x {L3}mm",
                    "Create a three-step shaft: {D1}x{L1}, {D2}x{L2}, {D3}x{L3} mm",
                    "Make a multi-diameter shaft with sections {D1}x{L1}mm, {D2}x{L2}mm, {D3}x{L3}mm",
                ],
                "params": {"D1": (15, 40), "L1": (20, 60), "D2": (10, 30), "L2": (20, 60), "D3": (5, 20), "L3": (15, 50)},
                "cad_template": {
                    "operations": [
                        {"type": "cylinder", "params": {"radius": "D1/2", "height": "L1"}},
                        {"type": "cylinder", "params": {"radius": "D2/2", "height": "L2"}},
                        {"type": "cylinder", "params": {"radius": "D3/2", "height": "L3"}}
                    ]
                }
            }
        ]

    def generate_sample(self, template: Dict) -> CADSample:
        text_template = random.choice(template["text_templates"])
        params = {}
        
        for key, (min_val, max_val) in template["params"].items():
            if isinstance(min_val, int):
                params[key] = random.randint(min_val, max_val)
            else:
                params[key] = round(random.uniform(min_val, max_val), 1)

        text = text_template.format(**params)
        
        cad_ops = []
        for op in template["cad_template"]["operations"]:
            op_params = {}
            for k, v in op["params"].items():
                if isinstance(v, str) and "/" in v:
                    num, denom = v.split("/")
                    op_params[k] = params[num] / float(denom)
                elif isinstance(v, str) and v in params:
                    op_params[k] = params[v]
                else:
                    op_params[k] = v
            cad_ops.append({"type": op["type"], "params": op_params})

        return CADSample(
            text_input=text,
            cad_output={"operations": cad_ops},
            metadata={"category": template["name"], "difficulty": "synthetic", "template": template["name"]}
        )

    def generate_dataset(self, num_samples: int) -> List[CADSample]:
        samples = []
        samples_per_template = num_samples // len(self.templates)
        remainder = num_samples % len(self.templates)

        for i, template in enumerate(self.templates):
            count = samples_per_template + (1 if i < remainder else 0)
            for _ in range(count):
                samples.append(self.generate_sample(template))

        random.shuffle(samples)
        return samples

def save_samples(samples: List[CADSample], filepath: str):
    data = [asdict(s) for s in samples]
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--output", type=str, default="data/synthetic_cad_samples.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generator = SyntheticDataGenerator(seed=args.seed)
    samples = generator.generate_dataset(args.num_samples)
    save_samples(samples, str(output_path))
    print(f"Generated {len(samples)} synthetic samples -> {output_path}")

    for i, s in enumerate(samples[:3]):
        print(f"\nSample {i+1}:")
        print(f"  Text: {s.text_input}")
        print(f"  CAD: {json.dumps(s.cad_output, indent=2)}")

if __name__ == "__main__":
    main()