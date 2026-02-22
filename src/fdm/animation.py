import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from IPython.display import HTML, display

mpl.rcParams["animation.embed_limit"] = 150

def ANI(x:np.ndarray, y:np.ndarray, U:np.ndarray, T:float, interval:int, title:str):
    
    Nx = len(x) - 1
    Ny = len(y) - 1

    Nt = U.shape[1] - 1

    dt = T / Nt
    
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(
        U[:,0].reshape((Nx+1, Ny+1)).T,
        cmap="turbo",
        interpolation="bilinear",
        origin="lower",
        extent=[x[0], x[-1], y[0], y[-1]],
        aspect="equal",
        vmin=np.min(U),
        vmax=np.max(U)
    )

    cbar = plt.colorbar(im, ax=ax, label=r"$u(x, y, t)$")

    cbar.set_ticks([])
    cbar.ax.tick_params(size=0)

    ax.set_title(title)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")

    ax.set_xticks([])
    ax.set_yticks([])

    time_text = ax.text(
        0.05, 0.95, "",
        transform=ax.transAxes,
        color="black",
        bbox=dict(
            facecolor="white",
            edgecolor="black",
            alpha=0.75
        )
    )
    
    def update(frame):
        im.set_data(U[:, frame].reshape((Nx+1, Ny+1)).T)
        time_text.set_text(f"t = {frame * dt:.3f}")
        return im, time_text
    
    ANI = animation.FuncAnimation(
        fig, update, frames=Nt+1, interval=interval, blit=False
    )
    
    plt.close(fig)

    display(HTML(ANI.to_jshtml()))
