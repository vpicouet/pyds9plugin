#%%
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz

# from astropy.io import fits
print(filename)
# header = fits.getheader("/Volumes/ExtremePro/sky_v0/test_az_el/stack27446457.fits")
# header = fits.getheader(
#     "/Volumes/ExtremePro/sky_v0/Stars_calib_frmes/stack27479757.fits"
# )
print("RADEC albireao =  +292.68646°/+27.9594°")
az, el, rot = header["AZ"], header["EL"], header["MROT"]
print("az,el,rot = ", az, el, rot)
time_header = header["DATE"]
FTS = EarthLocation(lat=34.404377 * u.deg, lon=104.193565 * u.deg, height=1400 * u.m)
# print("az el CNES:299.3, 29.96")
# print("good:299.3, 29.96")

utcoffset = -10 * u.hour  # Eastern Daylight Time
time = Time(time_header) - utcoffset
print(time)

altaz = SkyCoord(AltAz(obstime=time, az=az * u.deg, alt=el * u.deg, location=FTS))
# Transform back to RA, Dec;
# should get the same Dec as before but RA 12 hours different:
newradec = altaz.transform_to("icrs")
# print(m33.to_string("deg"))
print(newradec.to_string("decimal"))
# 5330'' en CE, et 615'' en EL biais DTu
# header["RA_AZALT"] = newradec.ra.deg
# header["DEC_AZAL"] = newradec.dec.deg

fits.setval(
    filename,
    "RA_AZALT",
    value=newradec.ra.deg,
    comment="From AZ ALT CNES info based on date/FTS",
)
fits.setval(
    filename,
    "DEC_AZAL",
    value=newradec.dec.deg,
    comment="From AZ ALT CNES info based on date/FTS",
)


#%%
# from astropy.io import fits
# header = fits.getheader(
#     "/Volumes/ExtremePro/sky_v0/Stars_calib_frmes/stack27509157.fits"
# )

ra, dec, rot = header["RA_DTU"], header["DEC_DTU"], header["ROLL_DTU"]


# def quat_multiply(self, quaternion0, quaternion1):
#     x0, y0, z0, w0 = np.split(quaternion0, 4, 1)
#     x1, y1, z1, w1 = np.split(quaternion1, 4, 1)

#     result = np.array(
#         (
#             x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
#             -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
#             x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
#             -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
#         ),
#         dtype=np.float64,
#     )
#     return np.transpose(np.squeeze(result))


# #%%
# def prod_quat(q=None, p=None):
#     # Cette fonction calcule le produit de deux quaternions
#     print(q, p)
#     # quat = np.array([[q(1),- q(2),- q(3),- q(4)],[q(2),q(1),- q(4),q(3)],[q(3),q(4),q(1),- q(2)],[q(4),- q(3),q(2),q(1)]]) * p
#     q2 = np.array(
#         [
#             [q[0], -q[1], -q[2], -q[3]],
#             [q[1], q[0], -q[3], q[2]],
#             [q[2], q[3], q[0], -q[1]],
#             [q[3], -q[2], q[1], q[0]],
#         ]
#     )
#     # quat = np.dot(q2, p)
#     print(q2, p)
#     quat = np.matmul(q2, p.T)
#     return quat


# def quat_to_mat(q):
#     q0 = q[0]
#     q1 = q[1]
#     q2 = q[2]
#     q3 = q[3]
#     M = np.array(
#         [
#             [
#                 2 * (q0 ** 2 + q1 ** 2) - 1,
#                 2 * (q1 * q2 + q0 * q3),
#                 2 * (q1 * q3 - q0 * q2),
#             ],
#             [
#                 2 * (q1 * q2 - q0 * q3),
#                 2 * (q0 ** 2 + q2 ** 2) - 1,
#                 2 * (q2 * q3 + q0 * q1),
#             ],
#             [
#                 2 * (q1 * q3 + q0 * q2),
#                 2 * (q2 * q3 - q0 * q1),
#                 2 * (q0 ** 2 + q3 ** 2) - 1,
#             ],
#         ]
#     )
#     return M


# def alpha_delta_rot(Q=None):
#     # Hypoth se:
#     # On suppose que l'axe de vis e sortant du SST a pour coordonn es [0 0 1]

#     # Entr e
#     # Q: quaternion entre le rep re inertiel et le rep re CU

#     # Sorties:
#     # alpha: ascension droite, entre 0 et 2*pi (rad)
#     # delta: d clinaison, entre -pi/2 et +pi/2 (rad)
#     # rot: rotation de champ, au sens du rep re inertiel, entre -pi et +pi (rad)

#     M = quat_to_mat(Q)
#     epsilon = 1e-06

#     print(M.shape, M[2, 2], M[2, 2].shape)
#     if np.abs(M[2, 2]) < 1 - epsilon:
#         delta = np.arcsin(M[2, 2])
#         alpha = np.arctan2(M[1, 2], M[0, 2])
#         rot = -np.arctan2(M[2, 1], M[2, 0])
#     else:
#         if M[2, 2] > 1 - epsilon:
#             delta = np.pi / 2
#             alpha = np.arctan2(M[1, 0], M[0, 0])
#             rot = 0
#         else:
#             delta = -np.pi / 2
#             alpha = np.arctan2(M[1, 0], M[0, 0])
#             rot = 0

#     if alpha < 0:
#         alpha = alpha + 2 * np.pi

#     return alpha, delta, rot


# def alpha_delta_rot_to_quat_CU(alpha=None, delta=None, rot=None):
#     # Entrées
#     # alpha : ascension droite (rad)
#     # delta : declinaison (rad)
#     # rot : rotation de champ (rad)

#     # Sorties
#     # q_I_CU: quaternion de passage du repère inertiel I au repère CU avec
#     # l'axe de visée Z pointant les coordonnées équatoriales (alpha,delta) et
#     # une rotation de champ finale (rot) autour de Z_CU
#     asc = alpha
#     dec = delta
#     q1 = np.transpose(np.array([[np.cos(asc / 2.0)], [0], [0], [np.sin(asc / 2.0)]]))
#     q2 = np.transpose(
#         np.array(
#             [
#                 [np.cos((np.pi / 2.0 - dec) / 2.0)],
#                 [0],
#                 [np.sin((np.pi / 2.0 - dec) / 2.0)],
#                 [0],
#             ]
#         )
#     )
#     q_I_CU_0 = prod_quat(q1[0], q2[0])
#     q3 = np.transpose(np.array([[np.cos(rot / 2.0)], [0], [0], [np.sin(rot / 2.0)]]))
#     q_I_CU = prod_quat(q_I_CU_0, q3)
#     return q_I_CU


# #%%


# #%%

# ##
# # quaternion DTU à 6h04h04 (image guider à 6h04:47 mais 50s d'avance pour
# # le guider)
# # mettre ici directement les biais envoyés par TC depuis le CC CNES
# import numpy as np

# biais_el = -615 / 3600 * np.pi / 180
# biais_ce = 5330 / 3600 * np.pi / 180
# biais_el = 100 / 3600 * np.pi / 180
# biais_ce = 0 / 3600 * np.pi / 180
# # biais_el =1.828 * np.pi / 180
# # biais_ce = 0.036 * np.pi / 180

# alpha_DTU = header["RA_DTU"] * np.pi / 180
# delta_DTU = header["DEC_DTU"] * np.pi / 180
# psi_DTU = header["ROLL_DTU"] * np.pi / 180
# dtu_quat = alpha_delta_rot_to_quat_CU(alpha_DTU, delta_DTU, psi_DTU).T[0][::-1]
# # dtu_quat = np.array([0.865754, 0.459764, 0.175396, 0.091233])
# # dtu_quat= [1 0 0 0];
# alpha, delta, psi = alpha_delta_rot(dtu_quat)
# alpha, psi = psi, alpha
# print(alpha_DTU, "=", alpha)
# print(delta_DTU, "=", delta)
# print(psi_DTU, "=", psi)

# dq = np.array([1, biais_ce / 2, biais_el / 2, 0])
# # dq = np.zeros((4))
# dq = dq / np.linalg.norm(dq)
# dtu_quat_corrige = prod_quat(dtu_quat, dq)
# alpha_corr, delta_corr, psi_corr = alpha_delta_rot(dtu_quat_corrige)

# print(
#     "Biais DTU appliqués : el=%2.8f arcsec, ce=%2.8f arcsec\n"
#     % (biais_el * 180 / np.pi * 3600, biais_ce * 180 / np.pi * 3600)
# )
# print(
#     "Q_DTU brut : alpha = %2.8f deg, delta = %2.8f deg, psi = %2.8f deg\n"
#     % (alpha * 180 / np.pi, delta * 180 / np.pi, psi * 180 / np.pi)
# )
# print(
#     "Q_DTU corrigé : alpha = %2.8f deg, delta = %2.8f deg, psi = %2.8f deg\n"
#     % (alpha_corr * 180 / np.pi, delta_corr * 180 / np.pi, psi_corr * 180 / np.pi)
# )



# import numpy as np

# mettre ici directement les biais envoyés par TC depuis le CC CNES

# (CAC) Affichage du cas test sous Matlab : 
#alpha_DTU=2.908521414101781e+02*pi/180;
#delta_DTU=28.906006074669364*pi/180;
#rot_DTU=71.297094252750597*pi/180;

#Biais DTU appliqués : el=-6580.80000000 arcsec, ce=-1209.60000000 arcsec
#Q_DTU brut : alpha = 290.85214141 deg, delta = 28.90600607 deg, rot = 71.29709425 deg
#Q_DTU corrigé : alpha = 292.69077522 deg, delta = 27.98918820 deg, rot = 70.42116608 deg

#%%
def quat_multiply(self, quaternion0, quaternion1):
    x0, y0, z0, w0 = np.split(quaternion0, 4, 1)
    x1, y1, z1, w1 = np.split(quaternion1, 4, 1)

    result = np.array(
        (
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
        ),
        dtype=np.float64,
    )
    return np.transpose(np.squeeze(result))

#%%
def prod_quat(q=None, p=None):
    # Cette fonction calcule le produit de deux quaternions
    print(q, p)
    # quat = np.array([[q(1),- q(2),- q(3),- q(4)],[q(2),q(1),- q(4),q(3)],[q(3),q(4),q(1),- q(2)],[q(4),- q(3),q(2),q(1)]]) * p
    q2 = np.array(
        [
            [q[0], -q[1], -q[2], -q[3]],
            [q[1], q[0], -q[3], q[2]],
            [q[2], q[3], q[0], -q[1]],
            [q[3], -q[2], q[1], q[0]],
        ]
    )
    # quat = np.dot(q2, p)
    print(q2, p)
    quat = np.matmul(q2, p.T)
    return quat

def quat_to_mat(q):
    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]
    M = np.array(
        [
            [
                2 * (q0 ** 2 + q1 ** 2) - 1,
                2 * (q1 * q2 + q0 * q3),
                2 * (q1 * q3 - q0 * q2),
            ],
            [
                2 * (q1 * q2 - q0 * q3),
                2 * (q0 ** 2 + q2 ** 2) - 1,
                2 * (q2 * q3 + q0 * q1),
            ],
            [
                2 * (q1 * q3 + q0 * q2),
                2 * (q2 * q3 - q0 * q1),
                2 * (q0 ** 2 + q3 ** 2) - 1,
            ],
        ]
    )
    return np.transpose(M)


def alpha_delta_rot(Q=None):
    # Hypoth se:
    # On suppose que l'axe de vis e sortant du SST a pour coordonn es [0 0 1]

    # Entr e
    # Q: quaternion entre le rep re inertiel et le rep re CU

    # Sorties:
    # alpha: ascension droite, entre 0 et 2*pi (rad)
    # delta: d clinaison, entre -pi/2 et +pi/2 (rad)
    # rot: rotation de champ, au sens du rep re inertiel, entre -pi et +pi (rad)

    M = quat_to_mat(Q)
    epsilon = 1e-06

    print(M.shape, M[2, 2], M[2, 2].shape)
    if np.abs(M[2, 2]) < 1 - epsilon:
        delta = np.arcsin(M[2, 2])
        alpha = np.arctan2(M[1, 2], M[0, 2])
        rot = -np.arctan2(M[2, 1], M[2, 0])
    else:
        if M[2, 2] > 1 - epsilon:
            delta = np.pi / 2
            alpha = np.arctan2(M[1, 0], M[0, 0])
            rot = 0
        else:
            delta = -np.pi / 2
            alpha = np.arctan2(M[1, 0], M[0, 0])
            rot = 0

    if alpha < 0:
        alpha = alpha + 2 * np.pi

    return alpha, delta, rot


def alpha_delta_rot_to_quat_CU(alpha=None, delta=None, rot=None):
    # Entrées
    # alpha : ascension droite (rad)
    # delta : declinaison (rad)
    # rot : rotation de champ (rad)

    # Sorties
    # q_I_CU: quaternion de passage du repère inertiel I au repère CU avec
    # l'axe de visée Z pointant les coordonnées équatoriales (alpha,delta) et
    # une rotation de champ finale (rot) autour de Z_CU
    asc = alpha
    dec = delta
    q1 = np.transpose(np.array([[np.cos(asc / 2.0)], [0], [0], [np.sin(asc / 2.0)]]))
    q2 = np.transpose(
        np.array(
            [
                [np.cos((np.pi / 2.0 - dec) / 2.0)],
                [0],
                [np.sin((np.pi / 2.0 - dec) / 2.0)],
                [0],
            ]
        )
    )
    q_I_CU_0 = prod_quat(q1[0], q2[0])
    q3 = np.transpose(np.array([[np.cos(rot / 2.0)], [0], [0], [np.sin(rot / 2.0)]]))
    q_I_CU = prod_quat(q_I_CU_0, q3)
    return q_I_CU

#%% Code de correction du quaternion DTU avec prise en compte des biais

biais_el = -1.828 * np.pi / 180
biais_ce = -0.336 * np.pi / 180

ra, dec, rot = header["RA_DTU"], header["DEC_DTU"], header["ROLL_DTU"]
# ra = 290.8521414101781
# dec = 28.906006074669364
# rot = 71.297094252750597

alpha_DTU = ra * np.pi / 180
delta_DTU = dec * np.pi / 180
rot_DTU = rot * np.pi / 180
dtu_quat = alpha_delta_rot_to_quat_CU(alpha_DTU, delta_DTU, rot_DTU).T[0] #[::-1]


dq = np.array([1, -biais_ce / 2, -biais_el / 2, 0])
dq = dq / np.linalg.norm(dq)

dtu_quat_corrige = prod_quat(dtu_quat, dq)
alpha_corr, delta_corr, rot_corr = alpha_delta_rot(dtu_quat_corrige)
rot_corr = np.arctan2(dtu_quat_corrige[2]*dtu_quat_corrige[3]+dtu_quat_corrige[0]*dtu_quat_corrige[1],-dtu_quat_corrige[1]*dtu_quat_corrige[3]+dtu_quat_corrige[0]*dtu_quat_corrige[2])

print(
    "Biais DTU appliqués : el=%2.8f arcsec, ce=%2.8f arcsec\n"
    % (biais_el * 180 / np.pi * 3600, biais_ce * 180 / np.pi * 3600)
)
print(
    "Q_DTU brut : alpha = %2.8f deg, delta = %2.8f deg, rot = %2.8f deg\n"
    % (ra, dec, rot)
)
print(
    "Q_DTU corrigé : alpha = %2.8f deg, delta = %2.8f deg, rot = %2.8f deg\n"
    % (alpha_corr * 180 / np.pi, delta_corr * 180 / np.pi, rot_corr * 180 / np.pi)
)


fits.setval(
    filename,
    "RADTU_C",
    value=alpha_corr * 180 / np.pi,
    comment="Offset corrected DTU information",
)
fits.setval(
    filename,
    "DECDTU_C",
    value=delta_corr * 180 / np.pi,
    comment="Offset corrected DTU information",
)

fits.setval(
    filename,
    "ROLDTU_C",
    value=psi_corr * 180 / np.pi,
    comment="Offset corrected DTU information",
)
