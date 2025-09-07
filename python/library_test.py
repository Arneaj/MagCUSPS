import time

t0 = time.time()
import sys
import importlib.util
from pathlib import Path

pkg_name = "topology_analysis"

# Check where Python is about to import the package from
spec = importlib.util.find_spec(pkg_name)
if spec and spec.origin:
    pkg_path = Path(spec.origin).resolve()
    site_packages_path = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    
    # If the package is coming from the source dir instead of site-packages, warn
    if str(pkg_path).startswith(str(Path.cwd())):
        raise RuntimeError(
            f"\n  You're running from inside the {pkg_name} source tree:\n"
            f"    {pkg_path}\n\n"
            f"This shadows the installed package in:\n"
            f"    {site_packages_path / pkg_name}\n\n"
            "Solution: run your script from outside the package folder "
            "or remove that path from sys.path."
        )

import numpy as np
import matplotlib.pyplot as plt
import mag_cusps as cusps
from gorgon_tools.magnetosphere import gorgon_import
import gorgon
import sys
t1 = time.time()
print(f"Finished in {t1-t0:.4f}s -> Modules loaded")




if len(sys.argv) < 2:
    print("No Run path given! Defaulting to /rds/general/user/avr24/projects/swimmr-sage/live/mheyns/benchmarking/runs/Run1")
    filepath = "/rds/general/user/avr24/projects/swimmr-sage/live/mheyns/benchmarking/runs/Run1"
else:
    filepath = sys.argv[1]


if len(sys.argv) < 3:
    print("No Timestep given! Defaulting to 23100")
    timestep = "23100"
else: 
    timestep = sys.argv[2]


if len(sys.argv) < 4 or sys.argv[3] == "xz":
    axis = 'xz'
elif sys.argv[3] == "xy":
    axis = 'xy'
else:
    print( "Please provide xy or xz" )




t0 = time.time()
sim = gorgon_import.gorgon_sim(data_dir=filepath)
index_of_timestep = np.where( sim.times == float(timestep) )[0][0]
sim.import_timestep(index_of_timestep)
sim.import_space( filepath + "/MS/x00_Bvec_c-" + timestep + ".pvtr" )

Rho: np.ndarray = sim.arr["rho"]
J: np.ndarray = sim.arr["jvec"]

X: np.ndarray = sim.xc; Y: np.ndarray = sim.yc; Z: np.ndarray = sim.zc
t1 = time.time()
print(f"Finished in {t1-t0:.4f}s -> Files read")


    





t0 = time.time()
extra_precision = 3.0

shape_realx2 = np.array([
    int( extra_precision * (X[-1]-X[0]) ), 
    int( extra_precision * (Y[-1]-Y[0]) ), 
    int( extra_precision * (Z[-1]-Z[0]) ),
    3
], dtype=np.int32)

# J_norm = np.linalg.norm( J, axis=3 )

J_processed: np.ndarray = cusps.preprocess( J, X, Y, Z, shape_realx2 )
Rho_processed: np.ndarray = cusps.preprocess( Rho, X, Y, Z, shape_realx2 )

J_norm_processed: np.ndarray = np.linalg.norm( J_processed, axis=3 )

earth_pos = extra_precision * np.array( [30, 58, 58], dtype=np.float64 )
t1 = time.time()
print(f"Finished in {t1-t0:.4f}s -> Preprocessing done")





t0 = time.time()
bs_radius = cusps.get_bowshock_radius( 0.0, 0.0, Rho_processed, earth_pos, 0.1 )
t1 = time.time()
print(f"Finished in {t1-t0:.4f}s -> Bowshock radius for (theta,phi) = (0.0, 0.0):", bs_radius)





t0 = time.time()
BS = cusps.get_bowshock( Rho_processed, earth_pos, 0.1, 32, 30 )
t1 = time.time()
print(f"Finished in {t1-t0:.4f}s -> Found entire Bowshock")



avg_std_dev = 0.0

t0 = time.time()
MP = cusps.get_interest_points(
    J_norm=J_norm_processed, 
    earth_pos=earth_pos, 
    Rho=Rho_processed,
    theta_min=0.0, 
    theta_max=np.pi*0.85,  
    nb_theta=40, 
    nb_phi=90,
    dx=0.1, 
    dr=0.1,
    alpha_0_min=0.4, alpha_0_max=0.6, nb_alpha_0=4,
    r_0_mult_min=1.5, r_0_mult_max=3.0, nb_r_0=20,
    avg_std_dev=avg_std_dev
)
t1 = time.time()
print(f"Finished in {t1-t0:.4f}s -> Found entire Magnetopause")




t0 = time.time()
MP_res = cusps.fit_to_analytical( 
    MP,                            # r_0                        a_0     a_1     a_2     d_n                     l_n     s_n     d_s                     l_s     s_s     e         
    initial_params      = np.array([ extra_precision * 10.0,    0.5,    0,      0,      extra_precision * 3,    0.55,   5,      extra_precision * 3,    0.55,   5,      0 ]),
    lowerbound          = np.array([ extra_precision * 5.0,     0.2,    -1.0,   -1.0,   extra_precision * 0,    0.1,    0.1,    extra_precision * 0,    0.1,    0.1,    -0.8 ]),
    upperbound          = np.array([ extra_precision * 15.0,    0.8,    1.0,    1.0,    extra_precision * 6,    2,      10,     extra_precision * 6,    2,      10,     0.8 ]),
    radii_of_variation  = np.array([ extra_precision * 3.0,     0.2,    0.5,    0.5,    extra_precision * 2,    0.1,    3,      extra_precision * 2,    0.1,    3,      0.5 ]),
    analytical_function = "Rolland25"
)
if MP_res is None:
    raise Exception()
MP_params, MP_cost = MP_res
t1 = time.time()
print(f"Finished in {t1-t0:.4f}s -> Fit Rolland25 function to the Magnetopause")
print( MP_params )


t0 = time.time()
params = np.concatenate([ np.array([MP_params[0]/extra_precision]), MP_params[1:4], np.array([MP_params[4]/extra_precision]), MP_params[5:7], np.array([MP_params[7]/extra_precision]), MP_params[8:] ])
analytics = cusps.analyse(
    J_norm = J_norm_processed, 
    earth_pos = earth_pos,
    nb_theta = 40, nb_phi = 90,
    interest_points = MP, 
    params = params, fit_loss = MP_cost,
    analytical_function = "Rolland25",
    threshold = extra_precision * 2.0
)

model = cusps.load_pretrained_model("Rolland25")
quality_score = model.predict(analytics)
uncertainty = model.get_sample_uncertainty(analytics)
t1 = time.time()
print(f"Finished in {t1-t0:.4f}s -> Analysis done:")
print("analytics =", analytics)
print("quality score =", quality_score)
print("uncertainty =", uncertainty)





t0 = time.time()
X_BS, Y_BS, Z_BS = gorgon.spherical_to_cartesian( BS[:,2], BS[:,0], BS[:,1], earth_pos )
X_MP, Y_MP, Z_MP = gorgon.spherical_to_cartesian( MP[:,2], MP[:,0], MP[:,1], earth_pos )

if axis == "xz":
    is_in_plane_BS = np.abs(Y_BS-int(earth_pos[1])) < 1

    X_BS_plot = X_BS[is_in_plane_BS]
    Z_BS_plot = Z_BS[is_in_plane_BS]

    is_in_plane_MP = np.abs(Y_MP-int(earth_pos[1])) < 1

    X_MP_plot = X_MP[is_in_plane_MP]
    Z_MP_plot = Z_MP[is_in_plane_MP]
    
    Theta = np.linspace(0, np.pi*0.99, 200)
    Phi = 0
    R1 = cusps.Rolland25( MP_params, Theta, Phi )
    X11, _, Z11 = gorgon.spherical_to_cartesian( R1, Theta, Phi, earth_pos )
    Phi = np.pi
    R1 = cusps.Rolland25( MP_params, Theta, Phi )
    X12, _, Z12 = gorgon.spherical_to_cartesian( R1, Theta, Phi, earth_pos )
    X1 = np.concatenate( [X12[::-1], X11] )
    Z1 = np.concatenate( [Z12[::-1], Z11] )
    
    X1_is_in = (X1 >= 0) * (X1 < J_norm_processed.shape[0])
    
    X1 = X1[X1_is_in]
    Z1 = Z1[X1_is_in]

    plt.imshow( np.moveaxis(Rho_processed[:,int(earth_pos[1]),:], [0,1], [1,0] ), cmap="inferno", norm="log" )
    # plt.imshow( np.moveaxis(J_norm_processed[:,int(earth_pos[1]),:], [0,1], [1,0] ), cmap="inferno", vmin=0, vmax=1e-9, interpolation="none")
    plt.colorbar()
    plt.scatter( X_BS_plot, Z_BS_plot, s=1.0, c="green" )
    plt.scatter( X_MP_plot, Z_MP_plot, s=1.0, c=MP[is_in_plane_MP,3] )
    plt.plot(X1, Z1)
    
    plt.ylabel(r"$z \in [-58; 58] R_E$")
    
else:
    is_in_plane_BS = np.abs(Z_BS-int(earth_pos[2])) < 1

    X_BS_plot = X_BS[is_in_plane_BS]
    Y_BS_plot = Y_BS[is_in_plane_BS]

    is_in_plane_MP = np.abs(Z_MP-int(earth_pos[2])) < 1

    X_MP_plot = X_MP[is_in_plane_MP]
    Y_MP_plot = Y_MP[is_in_plane_MP]
    
    Theta = np.linspace(0, np.pi*0.99, 200)
    Phi = np.pi/2
    R1 = cusps.Rolland25( MP_params, Theta, Phi )
    X21, Y21, _ = gorgon.spherical_to_cartesian( R1, Theta, Phi, earth_pos )
    Phi = -np.pi/2
    R1 = cusps.Rolland25( MP_params, Theta, Phi )
    X22, Y22, _ = gorgon.spherical_to_cartesian( R1, Theta, Phi, earth_pos )
    X2 = np.concatenate( [X22[::-1], X21] )
    Y2 = np.concatenate( [Y22[::-1], Y21] )
    
    X2_is_in = (X2 >= 0) * (X2 < J_norm_processed.shape[0])
    
    X2 = X2[X2_is_in]
    Y2 = Y2[X2_is_in]

    plt.imshow( np.moveaxis(Rho_processed[:,int(earth_pos[2]),:], [0,1], [1,0] ), cmap="inferno", norm="log" )
    # plt.imshow( np.moveaxis(J_norm_processed[:,:,int(earth_pos[2])], [0,1], [1,0] ), cmap="inferno", vmin=0, vmax=1e-9, interpolation="none")
    plt.colorbar()
    plt.scatter( X_BS_plot, Y_BS_plot, s=1.0, c="green" )
    plt.scatter( X_MP_plot, Y_MP_plot, s=1.0, c=MP[is_in_plane_MP,3] )
    plt.plot(X2, Y2)
    
    plt.ylabel(r"$y \in [-58; 58] R_E$")

plt.xlabel(r"$x \in [-30; 128] R_E$")


plt.savefig( "../images/bowshock.svg" )
t1 = time.time()
print(f"Finished in {t1-t0:.4f}s -> Saved the plot")
