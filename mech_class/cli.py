"""CLI for mech-class."""
from __future__ import annotations

from pathlib import Path

import click


@click.group()
@click.version_option()
def main() -> None:
    """mech-class: mechanism classifier for programmable genome-writing enzymes."""


@main.command()
@click.argument("fasta_file", type=click.Path(exists=True))
@click.option("--model-dir", "-m", type=click.Path(), required=True,
              help="Directory containing trained tier_a.pkl, tier_b.pkl, composite.pkl.")
@click.option("--output", "-o", default="predictions.parquet",
              help="Output Parquet file (default: predictions.parquet).")
@click.option("--device", default="cpu", help="Inference device: cpu or cuda.")
def predict(fasta_file: str, model_dir: str, output: str, device: str) -> None:
    """Predict mechanism class for all sequences in a FASTA file."""
    import pandas as pd
    from mech_class.api import Predictor

    click.echo(f"Loading models from {model_dir}...")
    predictor = Predictor.load(model_dir=Path(model_dir), device=device)

    click.echo(f"Running predictions on {fasta_file}...")
    results = predictor.predict_from_fasta(fasta_file)

    df = pd.DataFrame([r.model_dump() for r in results])
    df.to_parquet(output, compression="zstd")
    click.echo(f"Wrote {len(results)} predictions → {output}")

    # Summary
    click.echo("\nTier-A distribution:")
    click.echo(df["tier_a"].value_counts().to_string())
    n_composite = df["composite"].sum()
    click.echo(f"\nComposite architectures: {n_composite} / {len(df)}")


@main.command()
@click.argument("sequence")
@click.option("--accession", "-a", default="QUERY", help="Accession label.")
@click.option("--model-dir", "-m", type=click.Path(), required=True)
@click.option("--device", default="cpu")
def predict_one(sequence: str, accession: str, model_dir: str, device: str) -> None:
    """Predict mechanism for a single sequence string."""
    from mech_class.api import Predictor

    predictor = Predictor.load(model_dir=Path(model_dir), device=device)
    pred = predictor.predict_from_sequence(accession, sequence)

    click.echo(f"Accession:         {pred.accession}")
    click.echo(f"Tier A:            {pred.tier_a} (conf={pred.tier_a_confidence:.3f})")
    click.echo(f"Tier B:            {pred.tier_b} (conf={pred.tier_b_confidence:.3f})")
    click.echo(f"Composite:         {pred.composite}")
    if pred.composite_evidence:
        click.echo(f"Composite evidence: {', '.join(pred.composite_evidence)}")


if __name__ == "__main__":
    main()
