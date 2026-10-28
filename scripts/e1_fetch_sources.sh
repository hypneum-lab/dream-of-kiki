#!/bin/sh
# Fetch the three pinned E1 fact sources into DIR (section 5.1a, task 7).
# Verification is done by scripts/e1_corpus_build.py (RealTime QA by the
# git blob ids of the pinned commit; the CSVs by SHA-256 and size).
#
#   sh scripts/e1_fetch_sources.sh DIR
set -eu
DIR=${1:?usage: e1_fetch_sources.sh DIR}
RTQA_COMMIT=e288dcfe1d7818bf48299249ae892939637775db
DO_REV=455b35b2a2415034b05ef91e56c80191fef4a737
CF_REV=c25b89a968a2062e422dd96ddf0fe3387507f0dd
mkdir -p "$DIR/daily_oracle" "$DIR/mmlu_cf"
git clone -q --filter=blob:none --no-checkout \
  https://github.com/realtimeqa/realtimeqa_public.git "$DIR/realtimeqa"
git -C "$DIR/realtimeqa" checkout -q "$RTQA_COMMIT" -- past/2026
curl -sSfL -o "$DIR/daily_oracle/mc_questions_2020-01-01_2026-07-18.csv" \
  "https://huggingface.co/datasets/agentic-learning-ai-lab/daily-oracle/resolve/$DO_REV/mc_questions_2020-01-01_2026-07-18.csv"
curl -sSfL -o "$DIR/mmlu_cf/val.csv" \
  "https://huggingface.co/datasets/microsoft/MMLU-CF/resolve/$CF_REV/val.csv"
echo "fetched into $DIR; verify with scripts/e1_corpus_build.py --check or a build"
