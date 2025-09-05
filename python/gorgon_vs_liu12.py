import numpy as np
import matplotlib.pyplot as plt
import mag_cusps as cusps
from gorgon_tools.magnetosphere import gorgon_import
import gorgon
import sys

import sys

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
    
sim = gorgon_import.gorgon_sim(data_dir=filepath)
index_of_timestep = np.where( sim.times == float(timestep) )[0][0]
sim.import_timestep(index_of_timestep)
sim.import_space( filepath + "/MS/x00_Bvec_c-" + timestep + ".pvtr" )

Rho: np.ndarray = sim.arr["rho"]
J: np.ndarray = sim.arr["jvec"]

X: np.ndarray = sim.xc; Y: np.ndarray = sim.yc; Z: np.ndarray = sim.zc

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


MP = cusps.get_interest_points(
    J_norm_processed, earth_pos, 
    Rho_processed,
    theta_min=0.0, theta_max=np.pi*0.85,  
    nb_theta=60, nb_phi=90,
    dx=0.1, dr=0.1,
    alpha_0_min=0.4, alpha_0_max=0.6, nb_alpha_0=4,
    r_0_mult_min=1.5, r_0_mult_max=3.0, nb_r_0=20
)

if MP is None:
    raise Exception("Bad")

R25 = cusps.fit_to_analytical( 
    analytical_function = "Rolland25",
    interest_points     = MP,      # r_0                        a_0     a_1     a_2     d_n                     l_n     s_n     d_s                     l_s     s_s     e         
    initial_params      = np.array([ extra_precision * 10.0,    0.5,    0,      0,      extra_precision * 3,    0.55,   5,      extra_precision * 3,    0.55,   5,      0 ]),
    lowerbound          = np.array([ extra_precision * 5.0,     0.2,    -1.0,   -1.0,   extra_precision * 0,    0.1,    0.1,    extra_precision * 0,    0.1,    0.1,    -0.8 ]),
    upperbound          = np.array([ extra_precision * 15.0,    0.8,    1.0,    1.0,    extra_precision * 6,    2,      10,     extra_precision * 6,    2,      10,     0.8 ]),
    radii_of_variation  = np.array([ extra_precision * 3.0,     0.2,    0.5,    0.5,    extra_precision * 2,    0.1,    3,      extra_precision * 2,    0.1,    3,      0.5 ]),
)
if R25 is None:
    raise Exception("Bad bad")
R25_params, R25_cost = R25

L12 = cusps.fit_to_analytical( 
    analytical_function = "Liu12",
    interest_points     = MP,      # r_0                        a_0     a_1     a_2     d_n                     l_n     s_n     d_s                     l_s     s_s         
    initial_params      = np.array([ extra_precision * 10.0,    0.5,    0,      0,      extra_precision * 3,    0.55,   5,      extra_precision * 3,    0.55,   5]),
    lowerbound          = np.array([ extra_precision * 5.0,     0.2,    -1.0,   -1.0,   extra_precision * 0,    0.1,    0.1,    extra_precision * 0,    0.1,    0.1]),
    upperbound          = np.array([ extra_precision * 15.0,    0.8,    1.0,    1.0,    extra_precision * 6,    2,      10,     extra_precision * 6,    2,      10]),
    radii_of_variation  = np.array([ extra_precision * 3.0,     0.2,    0.5,    0.5,    extra_precision * 2,    0.1,    3,      extra_precision * 2,    0.1,    3]),
)
if L12 is None:
    raise Exception("Bad bad bad")
L12_params, L12_cost = L12



saturation = 1e-9



fig, axes = plt.subplots( 1, 2 )

axes[0].set_xlabel(r"$z \in [-58; 58] R_E$", fontsize=12, labelpad=6.0)
axes[0].set_ylabel(r"$x \in [-30; 128] R_E$", fontsize=12, labelpad=6.0)

axes[1].set_xlabel(r"$z \in [-58; 58] R_E$", fontsize=12, labelpad=6.0)
axes[1].set_ylabel(r"$x \in [-30; 128] R_E$", fontsize=12, labelpad=6.0)

fig.set_figwidth(9)
fig.set_figheight(6)

fig.suptitle(r"$||\mathbf{J}|| \in (\mathbf{P}_{\text{Earth}},\hat x, \hat z)$", fontsize=15)




theta = np.linspace(0, np.pi*0.99, 100)

################################
# LIU12
################################


r1 = cusps.Liu12( L12_params, theta, 0 )
r2 = cusps.Liu12( L12_params, theta, np.pi )


X1 = r1 * np.cos(theta)
Z1 = r1 * np.sin(theta)

X2 = r2 * np.cos(theta)
Z2 = r2 * np.sin(theta)

X = np.concatenate( [X2[::-1], X1] )
Z = np.concatenate( [-Z2[::-1], Z1] )

X += J_norm_processed.shape[0] - earth_pos[0]
Z += earth_pos[2]


axes[0].imshow( J_norm_processed[::-1,int(earth_pos[1]),:], cmap="inferno", vmin=0, vmax=saturation, interpolation="none")
axes[0].plot( Z, X )
axes[0].set_title( f"Liu12 model. Average fitting loss: {L12_cost/(extra_precision * MP.shape[0]):.2f}" )
axes[0].set_xlim(0, J_norm_processed.shape[2]-1)
axes[0].set_ylim(0, J_norm_processed.shape[0]-1)
axes[0].set_xticks([])
axes[0].set_yticks([])


############### ME25

r1 = cusps.Rolland25( R25_params, theta, 0 )
r2 = cusps.Rolland25( R25_params, theta, np.pi )

X1 = r1 * np.cos(theta)
Z1 = r1 * np.sin(theta)

X2 = r2 * np.cos(theta)
Z2 = r2 * np.sin(theta)

X = np.concatenate( [X2[::-1], X1] )
Z = np.concatenate( [-Z2[::-1], Z1] )

X += J_norm_processed.shape[0] - earth_pos[0]
Z += earth_pos[2]


axes[1].imshow( J_norm_processed[::-1,int(earth_pos[1]),:], cmap="inferno", vmin=0, vmax=saturation, interpolation="none")
# plt.colorbar(J1, ax=axes[1], shrink=1)
axes[1].plot( Z, X )
axes[1].set_title( f"New model. Average fitting loss: {R25_cost/(extra_precision * MP.shape[0]):.2f}" )
axes[1].set_xlim(0, J_norm_processed.shape[2]-1)
axes[1].set_ylim(0, J_norm_processed.shape[0]-1)
axes[1].set_xticks([])
axes[1].set_yticks([])






plt.savefig("../images/gorgon_vs_liu12.svg")
