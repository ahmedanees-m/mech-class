FROM pen-stack/plm:0.1.0

# SaProt 650M is loaded at runtime via HuggingFace hub.
# Mount ~/.cache/huggingface at runtime to avoid re-downloading:
#   -v ~/.cache/huggingface:/root/.cache/huggingface
# This keeps the image ~2.6 GB lighter.

# Foldseek — SA-token generation for SaProt input
RUN wget -q -O /tmp/foldseek.tar.gz \
        https://mmseqs.com/foldseek/foldseek-linux-avx2.tar.gz \
    && tar -xzf /tmp/foldseek.tar.gz -C /opt/ \
    && rm /tmp/foldseek.tar.gz \
    && ln -sf /opt/foldseek/bin/foldseek /usr/local/bin/foldseek

# DSSP for secondary structure
RUN apt-get update && apt-get install -y --no-install-recommends \
        dssp \
        wget \
    && rm -rf /var/lib/apt/lists/*

# fpocket for binding-pocket volume estimation
RUN apt-get update && apt-get install -y --no-install-recommends \
        fpocket \
    && rm -rf /var/lib/apt/lists/* \
    || (echo "[WARN] fpocket not in apt repos — building from source" && \
        apt-get install -y git cmake gcc g++ libnetcdf-dev && \
        git clone --depth 1 https://github.com/Discngine/fpocket.git /tmp/fpocket && \
        cd /tmp/fpocket && make && make install && rm -rf /tmp/fpocket)

# Python deps for structure feature extraction
RUN pip install --no-cache-dir \
    "biopython==1.83" \
    "pdb-tools==2.5.0" \
    "scipy==1.13.0" \
    "tqdm==4.66.4" \
    "duckdb>=0.10.0" \
    "pyarrow>=15.0.0" \
    "pandas>=2.2.0" \
    "requests>=2.31.0"

# Verify critical binaries are available
RUN foldseek version && \
    python -c "import Bio; print(f'BioPython {Bio.__version__} OK')" && \
    python -c "import torch; print(f'torch {torch.__version__} OK')"

CMD ["/bin/bash"]
