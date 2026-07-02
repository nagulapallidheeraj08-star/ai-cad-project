import json
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path

REQUIREMENTS_PATH = Path(__file__).resolve().with_name("requirements.txt")


def ensure_requirements() -> None:
    if not REQUIREMENTS_PATH.exists():
        return
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS_PATH)])


try:
    import cadquery as cq
    from cadquery import exporters
    import torch
    from torch.utils.data import Dataset
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
except ModuleNotFoundError as exc:
    ensure_requirements()
    try:
        import cadquery as cq
        from cadquery import exporters
        import torch
        from torch.utils.data import Dataset
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
    except ModuleNotFoundError as retry_exc:
        raise RuntimeError(
            f"Unable to import required package '{retry_exc.name}'. "
            f"Please run: {sys.executable} -m pip install -r {REQUIREMENTS_PATH}"
        ) from retry_exc


@dataclass
class CADSample:
    text_input: str
    cad_output: Dict[str, Any]
    metadata: Dict[str, Any]


class TextToCADDataset(Dataset):
    def __init__(self, samples: List[CADSample], tokenizer, max_length: int = 512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample.text_input
        cad_json = json.dumps(sample.cad_output)
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        target_encoding = self.tokenizer(
            cad_json,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': target_encoding['input_ids'].flatten()
        }


class CADGenerator:
    def __init__(self):
        self.operations = {
            'box': self._create_box,
            'cylinder': self._create_cylinder,
            'sphere': self._create_sphere,
            'hole': self._create_hole,
            'fillet': self._create_fillet,
            'chamfer': self._create_chamfer,
            'extrude': self._create_extrude,
            'revolve': self._create_revolve,
            'loft': self._create_loft,
            'sweep': self._create_sweep,
        }
    
    def _create_box(self, params: Dict) -> cq.Workplane:
        return cq.Workplane("XY").box(
            params.get('length', 10),
            params.get('width', 10),
            params.get('height', 10)
        )
    
    def _create_cylinder(self, params: Dict) -> cq.Workplane:
        return cq.Workplane("XY").circle(params.get('radius', 5)).extrude(params.get('height', 10))
    
    def _create_sphere(self, params: Dict) -> cq.Workplane:
        return cq.Workplane("XY").sphere(params.get('radius', 5))
    
    def _create_hole(self, params: Dict) -> cq.Workplane:
        wp = cq.Workplane("XY")
        if 'base' in params:
            wp = params['base']
        return wp.faces(">Z").workplane().hole(params.get('diameter', 5))
    
    def _create_fillet(self, params: Dict) -> cq.Workplane:
        wp = params.get('base', cq.Workplane("XY"))
        return wp.edges(params.get('edges', "|Z")).fillet(params.get('radius', 1))
    
    def _create_chamfer(self, params: Dict) -> cq.Workplane:
        wp = params.get('base', cq.Workplane("XY"))
        return wp.edges(params.get('edges', "|Z")).chamfer(params.get('distance', 1))
    
    def _create_extrude(self, params: Dict) -> cq.Workplane:
        wp = params.get('base', cq.Workplane("XY"))
        return wp.extrude(params.get('distance', 10))
    
    def _create_revolve(self, params: Dict) -> cq.Workplane:
        wp = params.get('base', cq.Workplane("XY"))
        return wp.revolve(params.get('angle', 360))
    
    def _create_loft(self, params: Dict) -> cq.Workplane:
        profiles = params.get('profiles', [])
        return cq.Workplane("XY").loft(profiles)
    
    def _create_sweep(self, params: Dict) -> cq.Workplane:
        path = params.get('path')
        profile = params.get('profile')
        return cq.Workplane("XY").sweep(path, profile)
    
    def generate_from_spec(self, spec: Dict) -> cq.Workplane:
        operations = spec.get('operations', [])
        result = None
        
        for op in operations:
            op_type = op.get('type')
            params = op.get('params', {})
            
            if op_type in self.operations:
                if result is None:
                    result = self.operations[op_type](params)
                else:
                    params['base'] = result
                    result = self.operations[op_type](params)
            else:
                raise ValueError(f"Unknown operation: {op_type}")
        
        return result
    
    def export_to_dict(self, shape: cq.Workplane) -> Dict:
        return {
            'type': 'cadquery',
            'bounds': shape.val().BoundingBox().__dict__ if shape.val() else {},
            'operations': []
        }


def create_training_samples() -> List[CADSample]:
    samples = [
        CADSample(
            text_input="Create a 50mm x 30mm x 20mm rectangular block with a 10mm diameter through-hole in the center",
            cad_output={
                'operations': [
                    {'type': 'box', 'params': {'length': 50, 'width': 30, 'height': 20}},
                    {'type': 'hole', 'params': {'diameter': 10}}
                ]
            },
            metadata={'category': 'basic_shapes', 'difficulty': 'easy'}
        ),
        CADSample(
            text_input="Design a cylindrical shaft 100mm long with 25mm diameter, with a 5mm keyway slot along its length",
            cad_output={
                'operations': [
                    {'type': 'cylinder', 'params': {'radius': 12.5, 'height': 100}},
                    {'type': 'box', 'params': {'length': 100, 'width': 5, 'height': 5}}
                ]
            },
            metadata={'category': 'mechanical_parts', 'difficulty': 'medium'}
        ),
        CADSample(
            text_input="Make a 40mm cube with 3mm fillets on all edges",
            cad_output={
                'operations': [
                    {'type': 'box', 'params': {'length': 40, 'width': 40, 'height': 40}},
                    {'type': 'fillet', 'params': {'radius': 3, 'edges': '|Z'}}
                ]
            },
            metadata={'category': 'basic_shapes', 'difficulty': 'easy'}
        ),
        CADSample(
            text_input="Create a hollow cylinder with 30mm outer diameter, 20mm inner diameter, and 50mm height",
            cad_output={
                'operations': [
                    {'type': 'cylinder', 'params': {'radius': 15, 'height': 50}},
                    {'type': 'cylinder', 'params': {'radius': 10, 'height': 50}}
                ]
            },
            metadata={'category': 'basic_shapes', 'difficulty': 'easy'}
        ),
        CADSample(
            text_input="Design a bracket: 60mm x 40mm base plate, 5mm thick, with two 8mm mounting holes 45mm apart, and a 30mm tall vertical flange with 3mm fillets",
            cad_output={
                'operations': [
                    {'type': 'box', 'params': {'length': 60, 'width': 40, 'height': 5}},
                    {'type': 'hole', 'params': {'diameter': 8}},
                    {'type': 'box', 'params': {'length': 60, 'width': 10, 'height': 30}},
                    {'type': 'fillet', 'params': {'radius': 3, 'edges': '|Z'}}
                ]
            },
            metadata={'category': 'mechanical_parts', 'difficulty': 'hard'}
        ),
        CADSample(
            text_input="Create a conical frustum: bottom radius 20mm, top radius 10mm, height 30mm",
            cad_output={
                'operations': [
                    {'type': 'cylinder', 'params': {'radius': 20, 'height': 30}},
                    {'type': 'cylinder', 'params': {'radius': 10, 'height': 30}}
                ]
            },
            metadata={'category': 'basic_shapes', 'difficulty': 'medium'}
        ),
    ]
    return samples


def save_samples(samples: List[CADSample], filepath: str):
    data = [asdict(s) for s in samples]
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_samples(filepath: str) -> List[CADSample]:
    with open(filepath, 'r') as f:
        data = json.load(f)
    return [CADSample(**item) for item in data]


def load_all_samples(data_dir: Path) -> List[CADSample]:
    all_samples = []
    for f in data_dir.glob("*.json"):
        try:
            samples = load_samples(str(f))
            all_samples.extend(samples)
            print(f"Loaded {len(samples)} samples from {f.name}")
        except Exception as e:
            print(f"Failed to load {f}: {e}")
    return all_samples


def setup_training(samples: List[CADSample], model_name: str = "Salesforce/codet5-small", output_dir: str = "./cad_model"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    dataset = TextToCADDataset(samples, tokenizer)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=2,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        learning_rate=3e-4,
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=4,
        dataloader_num_workers=2,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    
    return trainer


def generate_cad_from_text(text: str, model, tokenizer, cad_generator: CADGenerator) -> cq.Workplane:
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_length=512)
    cad_json = tokenizer.decode(outputs[0], skip_special_tokens=True)
    spec = json.loads(cad_json)
    return cad_generator.generate_from_spec(spec)


def main():
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)
    
    samples_file = data_dir / "cad_samples.json"
    
    if not samples_file.exists():
        samples = create_training_samples()
        save_samples(samples, str(samples_file))
        print(f"Created {len(samples)} training samples")
    else:
        samples = load_samples(str(samples_file))
        print(f"Loaded {len(samples)} training samples")
    
    # Load synthetic samples if available
    synthetic_files = list(data_dir.glob("synthetic_*.json"))
    if synthetic_files:
        synthetic_samples = []
        for f in synthetic_files:
            synthetic_samples.extend(load_samples(str(f)))
        samples.extend(synthetic_samples)
        print(f"Loaded {len(synthetic_samples)} synthetic samples")
        print(f"Total samples: {len(samples)}")
    
    for i, sample in enumerate(samples[:3]):
        print(f"\nSample {i+1}:")
        print(f"  Text: {sample.text_input}")
        print(f"  CAD: {json.dumps(sample.cad_output, indent=2)}")
    
    cad_generator = CADGenerator()
    for sample in samples[:2]:
        try:
            shape = cad_generator.generate_from_spec(sample.cad_output)
            export_path = data_dir / f"sample_{samples.index(sample)}.step"
            exporters.export(shape, str(export_path))
            print(f"Exported CAD to {export_path}")
        except Exception as e:
            print(f"Error generating CAD: {e}")
    
    print("\nTo start training, run:")
    print("  trainer = setup_training(samples)")
    print("  trainer.train()")


if __name__ == "__main__":
    main()