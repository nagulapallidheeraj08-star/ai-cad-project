import json
from text_to_cad import CADGenerator, create_training_samples, load_samples
from cadquery import exporters
from pathlib import Path

cad_gen = CADGenerator()

# Use the built-in samples (rule-based, works perfectly)
samples = create_training_samples()

# Also load synthetic samples if available
data_dir = Path("./data")
synthetic_file = data_dir / "synthetic_cad_samples.json"
if synthetic_file.exists():
    synthetic = load_samples(str(synthetic_file))
    samples.extend(synthetic)

print(f"Total samples available: {len(samples)}")

# Generate CAD for all samples
for i, sample in enumerate(samples[:10]):
    try:
        shape = cad_gen.generate_from_spec(sample.cad_output)
        if shape and shape.val():
            exporters.export(shape, f"output_{i}.step")
            print(f"[{i}] {sample.text_input}")
            print(f"    Saved: output_{i}.step")
        else:
            print(f"[{i}] Failed to generate shape")
    except Exception as e:
        print(f"[{i}] Error: {e}")

print("\nDone! Check output_0.step through output_9.step")
print("Open them in FreeCAD, Fusion 360, or any STEP viewer.")