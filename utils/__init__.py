"""Reusable utilities for the reinforcement-learning notebooks."""


def alert_training_complete(message="Training finished."):
    """Play a system notification sound and print a completion message."""
    try:
        import winsound

        winsound.MessageBeep(winsound.MB_ICONASTERISK)
    except (ImportError, RuntimeError):
        # Terminal bell fallback for platforms without winsound.
        print("\a", end="", flush=True)

    print(message)
