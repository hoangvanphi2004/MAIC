import warnings
import numpy as np

try:
    import imageio
except ImportError:
    imageio = None

def save_video(frames, vid_path):
    if imageio is None:
        warnings.warn('imageio not installed; cannot save video')
        return
    try:
        frames_u8 = [np.asarray(f).astype(np.uint8) for f in frames]
    except Exception:
        frames_u8 = frames
    try:
        imageio.mimsave(str(vid_path), frames_u8, fps=8)
    except Exception:
        try:
            with imageio.get_writer(str(vid_path), fps=8) as writer:
                for f in frames_u8:
                    writer.append_data(f)
        except Exception:
            pass
