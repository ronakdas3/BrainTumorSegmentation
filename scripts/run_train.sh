#!/usr/bin/env bash
# ==============================================================================
# Linux Server Training Runner with tmux
# ==============================================================================

SESSION_NAME="brats_swin"

echo "Checking for existing tmux session '${SESSION_NAME}'..."
tmux has-session -t ${SESSION_NAME} 2>/dev/null

if [ $? == 0 ]; then
    echo "Session '${SESSION_NAME}' already exists! Attaching..."
    tmux attach-session -t ${SESSION_NAME}
    exit 0
fi

echo "Creating new tmux session '${SESSION_NAME}' and launching training..."
tmux new-session -d -s ${SESSION_NAME}

# Send commands inside the session
tmux send-keys -t ${SESSION_NAME} "source activate monai || conda activate monai || true" C-m
tmux send-keys -t ${SESSION_NAME} "nvidia-smi" C-m
tmux send-keys -t ${SESSION_NAME} "python train.py --config configs/brats_swin_unetr.yaml | tee training.log" C-m

echo "---------------------------------------------------------------"
echo "Training started in detached tmux session '${SESSION_NAME}'!"
echo ""
echo "Useful commands:"
echo "  tmux attach -t ${SESSION_NAME}      # View live training output"
echo "  Press 'Ctrl+B' then 'D'             # Detach from tmux safely"
echo "  tail -f training.log                # Stream logs directly"
echo "---------------------------------------------------------------"
