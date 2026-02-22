import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.tri as tri

from IPython.display import HTML, display

mpl.rcParams["animation.embed_limit"] = 150

def ANI(X, Y, triangles, U, T, interval, title):
    
    Nt = U.shape[1] - 1
    dt = T / Nt

    triang = tri.Triangulation(X, Y, triangles)

    fig, ax = plt.subplots(figsize=(8, 6))

    vmax = np.max(np.abs(U))
    vmin = -vmax

    tpc = ax.tripcolor(
        triang, 
        U[:, 0],
        shading="gouraud",
        cmap="turbo",
        vmin=np.min(U),
        vmax=np.max(U)
    )

    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.set_aspect("equal")

    cbar = plt.colorbar(tpc, ax=ax, label=r"$u(x, y, t)$")

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
        tpc.set_array(U[:, frame]) 
        time_text.set_text(f"t = {frame * dt:.3f}")
        return tpc, time_text

    ANI = animation.FuncAnimation(
        fig, update, frames=Nt+1, interval=interval, blit=False
    )

    plt.close(fig)

    display(HTML(ANI.to_jshtml()))
    