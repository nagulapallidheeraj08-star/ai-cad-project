import json
import random
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from pathlib import Path
import math

@dataclass
class CADSample:
    text_input: str
    cad_output: Dict[str, Any]
    metadata: Dict[str, Any]

class EnhancedSyntheticDataGenerator:
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.templates = self._build_templates()
        self.parametric_families = self._build_parametric_families()

    def _build_templates(self) -> List[Dict]:
        return [
            {
                "name": "box_with_hole",
                "text_templates": [
                    "Create a {L}mm x {W}mm x {H}mm rectangular block with a {D}mm diameter through-hole in the center",
                    "Make a {L} by {W} by {H} box with a centered {D}mm hole through it",
                    "Design a {L}x{W}x{H}mm block featuring a {D}mm central bore",
                    "Generate a {L}mm long, {W}mm wide, {H}mm tall block with {D}mm hole",
                ],
                "params": {"L": (20, 150), "W": (15, 100), "H": (10, 80), "D": (3, 40)},
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
                "params": {"L": (50, 300), "D": (10, 80), "K": (3, 20)},
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
                "params": {"S": (20, 100), "R": (1, 15)},
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
                "params": {"OD": (20, 120), "ID": (10, 100), "H": (20, 200)},
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
                "params": {"L": (40, 150), "W": (30, 100), "T": (3, 15), "D": (5, 16), "S": (20, 100), "H": (20, 100), "R": (1, 8)},
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
                "params": {"R1": (15, 60), "R2": (5, 40), "H": (20, 100)},
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
                "params": {"L": (30, 100), "W": (30, 100), "H": (10, 40), "D1": (8, 25), "D2": (5, 20)},
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
                "params": {"D1": (10, 60), "L1": (30, 200), "D2": (30, 120), "T": (5, 30)},
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
                "params": {"L": (40, 150), "W": (40, 150), "H": (15, 60), "PL": (10, 80), "PW": (10, 80), "PH": (5, 30)},
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
                "params": {"L": (60, 200), "W": (60, 200), "T": (5, 20), "N": (2, 6), "M": (2, 6), "D": (4, 14), "S": (15, 50)},
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
                "params": {"D1": (15, 60), "L1": (20, 80), "D2": (10, 40), "L2": (20, 80), "D3": (5, 30), "L3": (15, 60)},
                "cad_template": {
                    "operations": [
                        {"type": "cylinder", "params": {"radius": "D1/2", "height": "L1"}},
                        {"type": "cylinder", "params": {"radius": "D2/2", "height": "L2"}},
                        {"type": "cylinder", "params": {"radius": "D3/2", "height": "L3"}}
                    ]
                }
            },
            {
                "name": "box_with_chamfer",
                "text_templates": [
                    "Create a {L}x{W}x{H}mm block with {C}mm chamfers on all top edges",
                    "Make a {L} by {W} by {H} box featuring {C}mm chamfers",
                    "Design a {L}x{W}x{H}mm block with {C}mm beveled edges",
                ],
                "params": {"L": (30, 100), "W": (30, 100), "H": (15, 50), "C": (1, 8)},
                "cad_template": {
                    "operations": [
                        {"type": "box", "params": {"length": "L", "width": "W", "height": "H"}},
                        {"type": "chamfer", "params": {"distance": "C", "edges": "|Z"}}
                    ]
                }
            },
            {
                "name": "cylinder_with_flange_and_holes",
                "text_templates": [
                    "Design a flanged cylinder {D1}mm x {L1}mm with {N} bolt holes {D2}mm diameter on {PCD}mm PCD",
                    "Create a pipe flange: {D1}mm bore, {L1}mm long, {N} x {D2}mm holes on {PCD}mm bolt circle",
                    "Make a {D1}x{L1}mm cylinder with {N}-hole flange pattern {D2}mm holes at {PCD}mm PCD",
                ],
                "params": {"D1": (20, 80), "L1": (20, 60), "N": (4, 12), "D2": (6, 16), "PCD": (50, 150)},
                "cad_template": {
                    "operations": [
                        {"type": "cylinder", "params": {"radius": "D1/2", "height": "L1"}},
                        {"type": "cylinder", "params": {"radius": "PCD/2", "height": "10"}},
                        {"type": "hole", "params": {"diameter": "D2"}}
                    ]
                }
            },
            {
                "name": "ribbed_plate",
                "text_templates": [
                    "Create a {L}x{W}x{T}mm plate with {N} reinforcing ribs {R_W}mm wide x {R_H}mm tall spaced {S}mm apart",
                    "Make a {L} by {W} by {T} base with {N} stiffening ribs {R_W}x{R_H}mm every {S}mm",
                    "Design a {L}x{W}x{T}mm plate featuring {N} ribs {R_W}mm wide, {R_H}mm high, {S}mm pitch",
                ],
                "params": {"L": (80, 200), "W": (60, 150), "T": (5, 15), "N": (3, 8), "R_W": (8, 20), "R_H": (10, 40), "S": (20, 50)},
                "cad_template": {
                    "operations": [
                        {"type": "box", "params": {"length": "L", "width": "W", "height": "T"}},
                        {"type": "box", "params": {"length": "R_W", "width": "W", "height": "R_H"}}
                    ]
                }
            },
            {
                "name": "tapered_shaft",
                "text_templates": [
                    "Design a tapered shaft: {D1}mm to {D2}mm over {L}mm length",
                    "Create a conical shaft from {D1}mm diameter to {D2}mm diameter, {L}mm long",
                    "Make a taper: {D1}mm base, {D2}mm tip, {L}mm length",
                ],
                "params": {"D1": (10, 50), "D2": (5, 30), "L": (50, 200)},
                "cad_template": {
                    "operations": [
                        {"type": "cylinder", "params": {"radius": "D1/2", "height": "L"}},
                        {"type": "cylinder", "params": {"radius": "D2/2", "height": "L"}}
                    ]
                }
            },
            {
                "name": "box_with_through_slot",
                "text_templates": [
                    "Create a {L}x{W}x{H}mm block with a {SL_W}mm wide x {SL_H}mm tall through slot centered",
                    "Make a {L} by {W} by {H} block with a {SL_W}x{SL_H}mm slot through the center",
                    "Design a {L}x{W}x{H}mm plate featuring a centered {SL_W}mm x {SL_H}mm slot",
                ],
                "params": {"L": (40, 150), "W": (40, 150), "H": (15, 50), "SL_W": (5, 30), "SL_H": (5, 30)},
                "cad_template": {
                    "operations": [
                        {"type": "box", "params": {"length": "L", "width": "W", "height": "H"}},
                        {"type": "box", "params": {"length": "SL_W", "width": "W", "height": "SL_H"}}
                    ]
                }
            },
            {
                "name": "hex_nut",
                "text_templates": [
                    "Design a hex nut: {AF}mm across flats, {T}mm thick, {D}mm thread diameter",
                    "Create a hexagonal nut {AF}mm AF x {T}mm thick for M{D} thread",
                    "Make a {AF}mm hex nut, {T}mm height, {D}mm hole",
                ],
                "params": {"AF": (8, 36), "T": (5, 20), "D": (4, 24)},
                "cad_template": {
                    "operations": [
                        {"type": "cylinder", "params": {"radius": "AF/2*1.155", "height": "T"}},
                        {"type": "hole", "params": {"diameter": "D"}}
                    ]
                }
            },
            {
                "name": "spherical_cap",
                "text_templates": [
                    "Create a spherical cap: radius {R}mm, height {H}mm",
                    "Make a sphere segment radius {R}mm, cap height {H}mm",
                    "Design a dome: {R}mm radius, {H}mm tall",
                ],
                "params": {"R": (15, 60), "H": (5, 40)},
                "cad_template": {
                    "operations": [
                        {"type": "sphere", "params": {"radius": "R"}},
                        {"type": "box", "params": {"length": "R*2", "width": "R*2", "height": "H"}}
                    ]
                }
            },
            {
                "name": "box_with_rounded_corners",
                "text_templates": [
                    "Create a {L}x{W}x{H}mm box with {R}mm radius rounded corners on all vertical edges",
                    "Make a {L} by {W} by {H} block with {R}mm corner fillets",
                    "Design a {L}x{W}x{H}mm rectangular prism with {R}mm rounded vertical corners",
                ],
                "params": {"L": (30, 120), "W": (30, 120), "H": (20, 80), "R": (2, 15)},
                "cad_template": {
                    "operations": [
                        {"type": "box", "params": {"length": "L", "width": "W", "height": "H"}},
                        {"type": "fillet", "params": {"radius": "R", "edges": "|Z"}}
                    ]
                }
            },
        ]

    def _build_parametric_families(self) -> List[Dict]:
        """Additional parametric families for more variety"""
        return [
            {
                "name": "gear_blank",
                "text_templates": [
                    "Design a gear blank: {OD}mm OD, {ID}mm bore, {T}mm thick, {N} teeth",
                    "Create a spur gear blank {OD}mm diameter, {ID}mm hole, {T}mm face width",
                    "Make a gear blank for {N} tooth gear: {OD}mm x {ID}mm x {T}mm",
                ],
                "params": {"OD": (30, 150), "ID": (8, 50), "T": (10, 40), "N": (12, 60)},
                "cad_template": {
                    "operations": [
                        {"type": "cylinder", "params": {"radius": "OD/2", "height": "T"}},
                        {"type": "hole", "params": {"diameter": "ID"}}
                    ]
                }
            },
            {
                "name": "enclosure_box",
                "text_templates": [
                    "Create an electronic enclosure: {L}x{W}x{H}mm outer, {WALL}mm wall thickness, {LID_T}mm lid thickness",
                    "Make a {L} by {W} by {H} box with {WALL}mm walls and {LID_T}mm lid",
                    "Design a project box {L}x{W}x{H}mm, {WALL}mm thick walls",
                ],
                "params": {"L": (60, 200), "W": (40, 150), "H": (30, 100), "WALL": (2, 5), "LID_T": (3, 6)},
                "cad_template": {
                    "operations": [
                        {"type": "box", "params": {"length": "L", "width": "W", "height": "H"}},
                        {"type": "box", "params": {"length": "L-2*WALL", "width": "W-2*WALL", "height": "H-WALL"}},
                        {"type": "box", "params": {"length": "L", "width": "W", "height": "LID_T"}}
                    ]
                }
            },
            {
                "name": "heat_sink",
                "text_templates": [
                    "Design a heat sink: {L}x{W}x{H}mm base with {N} fins {F_H}mm tall, {F_T}mm thick, {F_S}mm spacing",
                    "Create a {L} by {W} by {H} heat sink with {N} fins {F_H}x{F_T}mm every {F_S}mm",
                    "Make a heat sink base {L}x{W}x{H}mm, {N} fins {F_H}mm high, {F_T}mm thick, {F_S}mm pitch",
                ],
                "params": {"L": (40, 120), "W": (40, 120), "H": (5, 15), "N": (5, 20), "F_H": (15, 50), "F_T": (1.5, 4), "F_S": (3, 10)},
                "cad_template": {
                    "operations": [
                        {"type": "box", "params": {"length": "L", "width": "W", "height": "H"}},
                        {"type": "box", "params": {"length": "F_T", "width": "W", "height": "F_H"}}
                    ]
                }
            },
        ]

    def generate_sample(self, template: Dict) -> CADSample:
        text_template = random.choice(template["text_templates"])
        params = {}
        
        for key, (min_val, max_val) in template["params"].items():
            if isinstance(min_val, int):
                params[key] = random.randint(min_val, max_val)
            else:
                params[key] = round(random.uniform(min_val, max_val), 1)

        # Ensure derived params make sense
        if "ID" in params and "OD" in params and params["ID"] >= params["OD"]:
            params["ID"] = max(params["OD"] - 5, 1)
        if "R2" in params and "R1" in params and params["R2"] >= params["R1"]:
            params["R2"] = params["R1"] - 1
        if "D2" in params and "D1" in params and params["D2"] >= params["D1"]:
            params["D2"] = max(params["D1"] - 3, 1)

        text = text_template.format(**params)
        
        cad_ops = []
        for op in template["cad_template"]["operations"]:
            op_params = {}
            for k, v in op["params"].items():
                if isinstance(v, str):
                    try:
                        # Safe eval with params namespace
                        op_params[k] = eval(v, {"__builtins__": {}}, params)
                    except:
                        op_params[k] = v
                else:
                    op_params[k] = v
            cad_ops.append({"type": op["type"], "params": op_params})

        return CADSample(
            text_input=text,
            cad_output={"operations": cad_ops},
            metadata={"category": template["name"], "difficulty": "synthetic", "template": template["name"]}
        )

    def generate_dataset(self, num_samples: int) -> List[CADSample]:
        all_templates = self.templates + self.parametric_families
        samples = []
        samples_per_template = num_samples // len(all_templates)
        remainder = num_samples % len(all_templates)

        for i, template in enumerate(all_templates):
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
    parser.add_argument("--num-samples", type=int, default=100000)
    parser.add_argument("--output", type=str, default="data/enhanced_cad_samples.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generator = EnhancedSyntheticDataGenerator(seed=args.seed)
    samples = generator.generate_dataset(args.num_samples)
    save_samples(samples, str(output_path))
    print(f"Generated {len(samples)} enhanced samples -> {output_path}")

    for i, s in enumerate(samples[:5]):
        print(f"\nSample {i+1}:")
        print(f"  Text: {s.text_input}")
        print(f"  CAD: {json.dumps(s.cad_output, indent=2)}")

if __name__ == "__main__":
    main()