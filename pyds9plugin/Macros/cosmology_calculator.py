from unittest.loader import VALID_MODULE_NAME


def cosmology_calculator(xpapoint=None, argv=[]):
    """Plot the different information for a given cosmology
    """
    # from dataphile.graphics.widgets import Slider

    from matplotlib.widgets import Slider
    import numpy as np


    cosmology, redshift, H0, Omega_m, Ode0, uncertainty = (
        "LambdaCDM",
        "0.7",
        70,
        0.30,
        0.7,
        "H0:1",
    )
    redshift = np.array(redshift.split("-"), dtype=float)
    if cosmology == "w0waCDM":
        from astropy.cosmology import w0waCDM as cosmol
    elif cosmology == "w0wzCDM":
        from astropy.cosmology import w0wzCDM as cosmol
    elif cosmology == "wpwaCDM":
        from astropy.cosmology import wpwaCDM as cosmol
    elif cosmology == "LambdaCDM":
        from astropy.cosmology import LambdaCDM as cosmol
    elif cosmology == "wCDM":
        from astropy.cosmology import wCDM as cosmol

    if cosmology == "WMAP9":
        from astropy.cosmology import WMAP9 as cosmo
    elif cosmology == "WMAP7":
        from astropy.cosmology import WMAP7 as cosmo
    elif cosmology == "WMAP5":
        from astropy.cosmology import WMAP5 as cosmo
    elif cosmology == "Planck13":
        from astropy.cosmology import Planck13 as cosmo
    elif cosmology == "Planck15":
        from astropy.cosmology import Planck15 as cosmo

    elif (cosmology == "wCDM") or (cosmology == "LambdaCDM"):
        H0, Omega_m, Ode0 = np.array([H0, Omega_m, Ode0], dtype=float)
        param, uncertainty = uncertainty.split(":")
        uncertainty = float(uncertainty)
        cosmo = cosmol(H0=H0, Om0=Omega_m, Ode0=Ode0)
        #verboseprint("param, uncertainty = ", param, uncertainty)
        if param.lower() == "h0":
            cosmo1 = cosmol(H0=H0 * (1 - 0.01 * uncertainty), Om0=Omega_m, Ode0=Ode0)
            cosmo2 = cosmol(H0=H0 * (1 + 0.01 * uncertainty), Om0=Omega_m, Ode0=Ode0)
        elif param.lower() == "om0":
            cosmo1 = cosmol(H0=H0, Om0=Omega_m * (1 - 0.01 * uncertainty), Ode0=Ode0)
            cosmo2 = cosmol(H0=H0, Om0=Omega_m * (1 + 0.01 * uncertainty), Ode0=Ode0)
        else:
            cosmo1 = cosmol(H0=H0, Om0=Omega_m, Ode0=Ode0 * (1 - 0.01 * uncertainty))
            cosmo2 = cosmol(H0=H0, Om0=Omega_m, Ode0=Ode0 * (1 + 0.01 * uncertainty))
    elif cosmology == "default_cosmology":
        from astropy.cosmology import default_cosmology

        cosmo = default_cosmology.get()
        cosmo1 = cosmo2 = cosmo
        #verboseprint(cosmo1)
        #verboseprint(cosmo2)
    # TODO make it quicker
    info = {}
    info["luminosity_distance"] = cosmo.luminosity_distance(redshift)
    info["age"] = cosmo.age(redshift)
    info["kpc_proper_per_arcsec"] = 1 / cosmo.arcsec_per_kpc_proper(redshift)
    info["kpc_comoving_per_arcsec"] = 1 / cosmo.arcsec_per_kpc_comoving(redshift)
    info["arcsec_per_proper_kpc"] = cosmo.arcsec_per_kpc_proper(redshift)
    info["arcsec_per_comoving_kpc"] = cosmo.arcsec_per_kpc_comoving(redshift)
    info["angular_diameter_distance"] = cosmo.angular_diameter_distance(redshift)
    info["comoving_distance"] = cosmo.comoving_distance(redshift)
    info["comoving_volume"] = cosmo.comoving_volume(redshift)
    info["lookback_distance"] = cosmo.lookback_distance(redshift)
    info["lookback_time"] = cosmo.lookback_time(redshift)
    info["scale_factor"] = cosmo.scale_factor(redshift)
    info["efunc"] = cosmo.efunc(redshift)

    zs = np.linspace(0, 5, 50)

    if type(redshift) is float:
        redshifts = np.array([redshift], dtype=float)
    else:
        redshifts = np.array(redshift, dtype=float)
    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3) = plt.subplots(3, 3, figsize=(18, 9.5), sharex=True)
    a = 0.08
    redshift_ = Slider(
        figure=fig,
        # location=[0.1, 0.14 - a, 0.8, 0.03],
        ax=plt.axes([0.1, 0.14 - a, 0.8, 0.03], facecolor="None"),
        valmin=0,
        valmax = 6, 
        label="z",
        # bounds=(0, 5),
        # value=redshift,
        # init_value=redshift,
    )
    H0_ = Slider(
        figure=fig,
        # location=[0.1, 0.12 - a, 0.8, 0.03],
        label="$H_0$",
        # bounds=(0, 100),
        # init_value=H0,
        ax=plt.axes([0.1, 0.12 - a, 0.8, 0.03], facecolor="None"),
        valmin=0,
        valmax=100,
    )
    Omega_m_ = Slider(
        figure=fig,
        # location=[0.1, 0.10 - a, 0.8, 0.03],
        label=r"$\Omega_m$",
        # bounds=(0, 1),
        # init_value=Omega_m,
        ax=plt.axes([0.1, 0.10 - a, 0.8, 0.03], facecolor="None"),
        valmin=0,
        valmax = 1, 
    )
    Ode0_ = Slider(
        figure=fig,
        # location=[0.1, 0.08 - a, 0.8, 0.03],
        label="$Ode_0$",
        # bounds=(0, 1),
        # init_value=Ode0,
        ax=plt.axes([0.1, 0.08 - a, 0.8, 0.03], facecolor="None"),
        valmin=0,
        valmax = 1, 
    )

    t = "U4"
    l = " - "
    p = ax1[0].plot(
        zs,
        cosmo.angular_diameter_distance(zs) / 1000,
        label="Angular diameter distance = %s"
        % (
            l.join(
                np.array(
                    cosmo.angular_diameter_distance(redshifts).value / 1000, dtype=t,
                )
            )
        ),
    )
    ax1[0].set_ylabel("Gpc")
    ax1[0].legend(loc="upper left")
    for redshift in redshifts:
        a10 = ax1[0].plot(
            redshift * np.ones(2),
            [0, (cosmo.angular_diameter_distance(redshift) / 1000).value],
            linestyle="dotted",
            color=p[0].get_color(),
            label="_nolegend_",
        )
        a10v = ax1[0].plot(
            [0, redshift],
            np.ones(2) * (cosmo.angular_diameter_distance(redshift) / 1000).value,
            linestyle="dotted",
            color=p[0].get_color(),
            label="_nolegend_",
        )
    p10 = ax1[0].plot(
        zs,
        cosmo.comoving_distance(zs) / 1000,
        label="Comoving distance = %s"
        % (l.join(np.array(cosmo.comoving_distance(redshifts).value / 1000, dtype=t))),
    )
    for redshift in redshifts:
        a10_ = ax1[0].plot(
            redshift * np.ones(2),
            [0, (cosmo.comoving_distance(redshift) / 1000).value],
            linestyle="dotted",
            color=p10[0].get_color(),
            label="_nolegend_",
        )
        a10v_ = ax1[0].plot(
            [0, redshift],
            np.ones(2) * (cosmo.comoving_distance(redshift) / 1000).value,
            linestyle="dotted",
            color=p10[0].get_color(),
            label="_nolegend_",
        )

    p11 = ax1[1].plot(
        zs,
        cosmo.luminosity_distance(zs) / 1000,
        label="Luminosity distance = %s"
        % (
            l.join(np.array(cosmo.luminosity_distance(redshifts).value / 1000, dtype=t))
        ),
    )
    ax1[1].set_ylabel("Gpc")
    ax1[1].legend(loc="upper left")
    for redshift in redshifts:
        a11 = ax1[1].plot(
            redshift * np.ones(2),
            [0, (cosmo.luminosity_distance(redshift) / 1000).value],
            linestyle="dotted",
            color=p[0].get_color(),
            label="_nolegend_",
        )
        a11v = ax1[1].plot(
            [0, redshift],
            np.ones(2) * (cosmo.luminosity_distance(redshift) / 1000).value,
            linestyle="dotted",
            color=p[0].get_color(),
            label="_nolegend_",
        )

    p12 = ax1[2].plot(
        zs,
        cosmo.critical_density(zs) / 1e-29,
        label="Critical density = %s"
        % (l.join(np.array(cosmo.critical_density(redshifts).value / 1e-29, dtype=t))),
    )
    ax1[2].set_ylabel("10e-29 g/cm^3")
    ax1[2].legend(loc="upper left")
    for redshift in redshifts:
        a12 = ax1[2].plot(
            redshift * np.ones(2),
            [0, (cosmo.critical_density(redshift) / 1e-29).value],
            linestyle="dotted",
            color=p[0].get_color(),
            label="_nolegend_",
        )
        a12v = ax1[2].plot(
            [0, redshift],
            np.ones(2) * (cosmo.critical_density(redshift) / 1e-29).value,
            linestyle="dotted",
            color=p[0].get_color(),
            label="_nolegend_",
        )

    p20 = ax2[0].plot(
        zs,
        cosmo.comoving_volume(zs) / 1e9,
        label="Comoving volume = %s"
        % (l.join(np.array(cosmo.comoving_volume(redshifts).value / 1e9, dtype=t))),
    )
    ax2[0].set_ylabel("Gpc^3")
    ax2[0].legend(loc="upper left")
    for redshift in redshifts:
        a20 = ax2[0].plot(
            redshift * np.ones(2),
            [0, (cosmo.comoving_volume(redshift) / 1e9).value],
            linestyle="dotted",
            color=p[0].get_color(),
            label="_nolegend_",
        )
        a20v = ax2[0].plot(
            [0, redshift],
            np.ones(2) * (cosmo.comoving_volume(redshift) / 1e9).value,
            linestyle="dotted",
            color=p[0].get_color(),
            label="_nolegend_",
        )

    p21 = ax2[1].plot(
        zs,
        cosmo.lookback_time(zs),
        label="Lookback time = %s"
        % (l.join(np.array(cosmo.lookback_time(redshifts).value, dtype=t))),
    )
    for redshift in redshifts:
        a21 = ax2[1].plot(
            redshift * np.ones(2),
            [0, (cosmo.lookback_time(redshift)).value],
            linestyle="dotted",
            color=p[0].get_color(),
            label="_nolegend_",
        )
        a21v = ax2[1].plot(
            [0, redshift],
            np.ones(2) * (cosmo.lookback_time(redshift)).value,
            linestyle="dotted",
            color=p[0].get_color(),
            label="_nolegend_",
        )

    p21_ = ax2[1].plot(
        zs,
        cosmo.age(zs),
        label="age = %s" % (l.join(np.array(cosmo.age(redshifts).value, dtype=t))),
    )
    ax2[1].legend(loc="upper left")
    ax2[1].set_ylabel("Gyr")
    for redshift in redshifts:
        a21_ = ax2[1].plot(
            redshift * np.ones(2),
            [0, (cosmo.age(redshift)).value],
            linestyle="dotted",
            color=p21_[0].get_color(),
            label="_nolegend_",
        )
        a21v_ = ax2[1].plot(
            [0, redshift],
            np.ones(2) * (cosmo.age(redshift)).value,
            linestyle="dotted",
            color=p21_[0].get_color(),
            label="_nolegend_",
        )

    p22 = ax2[2].plot(
        zs,
        cosmo.distmod(zs),
        label="Dist mod (mu) = %s"
        % (l.join(np.array(cosmo.distmod(redshifts).value, dtype=t))),
    )
    ax2[2].legend()
    ax2[2].set_ylabel("mag")
    for redshift in redshifts:
        a22 = ax2[2].plot(
            redshift * np.ones(2),
            [0, (cosmo.distmod(redshift)).value],
            linestyle="dotted",
            color=p[0].get_color(),
            label="_nolegend_",
        )
        a22v = ax2[2].plot(
            [0, redshift],
            np.ones(2) * (cosmo.distmod(redshift)).value,
            linestyle="dotted",
            color=p[0].get_color(),
            label="_nolegend_",
        )

    p30 = ax3[0].plot(
        zs,
        cosmo.efunc(zs),
        label="efunc = %s" % (l.join(np.array(cosmo.efunc(redshifts), dtype=t))),
    )
    ax3[0].set_ylabel("E(z)")
    ax3[0].legend(loc="upper left")
    for redshift in redshifts:
        a30 = ax3[0].plot(
            redshift * np.ones(2),
            [0, (cosmo.efunc(redshift))],
            linestyle="dotted",
            color=p[0].get_color(),
            label="_nolegend_",
        )
        a30v = ax3[0].plot(
            [0, redshift],
            np.ones(2) * (cosmo.efunc(redshift)),
            linestyle="dotted",
            color=p[0].get_color(),
            label="_nolegend_",
        )

    p31 = ax3[1].plot(
        zs,
        cosmo.scale_factor(zs),
        label="Scale factor = %s"
        % (l.join(np.array(cosmo.scale_factor(redshifts), dtype=t))),
    )
    ax3[1].legend(loc="upper left")
    ax3[1].set_xlabel("Redshift")
    ax3[1].set_ylabel("a")
    for redshift in redshifts:
        a31 = ax3[1].plot(
            redshift * np.ones(2),
            [0, (cosmo.scale_factor(redshift))],
            linestyle="dotted",
            color=p[0].get_color(),
            label="_nolegend_",
        )
        a31v = ax3[1].plot(
            [0, redshift],
            np.ones(2) * (cosmo.scale_factor(redshift)),
            linestyle="dotted",
            color=p[0].get_color(),
            label="_nolegend_",
        )

    p32 = ax3[2].plot(
        zs,
        1 / cosmo.arcsec_per_kpc_proper(zs),
        label="Proper = %s"
        % (l.join(np.array(1 / cosmo.arcsec_per_kpc_proper(redshifts).value, dtype=t))),
    )
    for redshift in redshifts:
        a32 = ax3[2].plot(
            redshift * np.ones(2),
            [0, 1 / (cosmo.arcsec_per_kpc_proper(redshift)).value],
            linestyle="dotted",
            color=p[0].get_color(),
            label="_nolegend_",
        )
        a32v = ax3[2].plot(
            [0, redshift],
            np.ones(2) * 1 / (cosmo.arcsec_per_kpc_proper(redshift)).value,
            linestyle="dotted",
            color=p[0].get_color(),
            label="_nolegend_",
        )
    p32_ = ax3[2].plot(
        zs,
        1 / cosmo.arcsec_per_kpc_comoving(zs),
        label="Comoving = %s"
        % (
            l.join(
                np.array(1 / cosmo.arcsec_per_kpc_comoving(redshifts).value, dtype=t)
            )
        ),
    )
    ax3[2].legend(loc="upper left")
    ax3[2].set_ylabel("'/kpc")
    for redshift in redshifts:
        a32_ = ax3[2].plot(
            redshift * np.ones(2),
            [0, 1 / (cosmo.arcsec_per_kpc_comoving(redshift)).value],
            linestyle="dotted",
            color=p32_[0].get_color(),
            label="_nolegend_",
        )
        a32v_ = ax3[2].plot(
            [0, redshift],
            np.ones(2) * 1 / (cosmo.arcsec_per_kpc_comoving(redshift)).value,
            linestyle="dotted",
            color=p32_[0].get_color(),
            label="_nolegend_",
        )

    dict_ = {"redshift": redshift, "H0": H0, "Omega_m": Omega_m, "Ode0": Ode0}

    def update(val):
        # dict_["redshift"] = redshift_.value
        # dict_["H0"] = H0_.value
        # dict_["Omega_m"] = Omega_m_.value
        # dict_["Ode0"] = Ode0_.value
        # redshift = redshift_.value

        dict_["redshift"] = redshift_.val
        dict_["H0"] = H0_.val
        dict_["Omega_m"] = Omega_m_.val
        dict_["Ode0"] = Ode0_.val
        redshift = redshift_.val

        from astropy.cosmology import LambdaCDM as cosmol

        cosmo = cosmol(H0=dict_["H0"], Om0=dict_["Omega_m"], Ode0=dict_["Ode0"])
        p[0].set_ydata(cosmo.angular_diameter_distance(zs) / 1000)
        p10[0].set_ydata(cosmo.comoving_distance(zs) / 1000)
        p11[0].set_ydata(cosmo.luminosity_distance(zs) / 1000)
        p12[0].set_ydata(cosmo.critical_density(zs) / 1e-29)
        p20[0].set_ydata(cosmo.comoving_volume(zs) / 1e9)
        p21[0].set_ydata(cosmo.lookback_time(zs))
        p21_[0].set_ydata(cosmo.age(zs))
        p22[0].set_ydata(cosmo.distmod(zs))
        p30[0].set_ydata(cosmo.efunc(zs))
        p31[0].set_ydata(cosmo.scale_factor(zs))
        p32[0].set_ydata(1 / cosmo.arcsec_per_kpc_proper(zs))
        p32_[0].set_ydata(1 / cosmo.arcsec_per_kpc_comoving(zs))
        ps = [p, p10, p11, p12, p20, p21, p21_, p22, p30, p31, p32, p32_]
        a_ = [a10, a10_, a11, a12, a20, a21, a21_, a22, a30, a31, a32, a32_]
        av_ = [
            a10v,
            a10v_,
            a11v,
            a12v,
            a20v,
            a21v,
            a21v_,
            a22v,
            a30v,
            a31v,
            a32v,
            a32v_,
        ]
        im_ = [
            (cosmo.angular_diameter_distance(redshift) / 1000).value,
            (cosmo.comoving_distance(redshift) / 1000).value,
            (cosmo.luminosity_distance(redshift) / 1000).value,
            (cosmo.critical_density(redshift) / 1e-29).value,
            (cosmo.comoving_volume(redshift) / 1e9).value,
            (cosmo.lookback_time(redshift)).value,
            (cosmo.age(redshift)).value,
            (cosmo.distmod(redshift)).value,
            cosmo.efunc(redshift),
            cosmo.scale_factor(redshift),
            1 / (cosmo.arcsec_per_kpc_proper(redshift)).value,
            1 / (cosmo.arcsec_per_kpc_comoving(redshift)).value,
        ]
        legends = [
            "Angular diameter distance = %s",
            "Comoving distance = %s",
            "Luminosity distance = %s",
            "Critical density = %s",
            "Comoving volume = %s",
            "Lookback time = %s",
            "age = %s",
            "Dist mod (mu) = %s",
            "efunc = %s",
            "Scale factor = %s",
            "Proper = %s",
            "Comoving = %s",
        ]
        for i, (a, b, c, legendi, psi) in enumerate(zip(a_, av_, im_, legends, ps)):
            a[0].set_xdata(redshift * np.ones(2))
            a[0].set_ydata([0, c])
            b[0].set_xdata([0, redshift])
            b[0].set_ydata(np.ones(2) * (c))
            psi[0].set_label(legendi % (np.around(c, 2)))
        for ax in np.array([ax1, ax2, ax3]).ravel():
            ax.legend(loc="upper left")
        fig.canvas.draw_idle()

    redshift_.on_changed(update)
    H0_.on_changed(update)
    Omega_m_.on_changed(update)
    Ode0_.on_changed(update)

    # #verboseprint("%s : H0=%s, Om0=%s, Ode0=%s, Tcmb0=%s, Neff=%s, Ob0=%s"
    #     % (
    #         cosmology,
    #         cosmo.H0,
    #         cosmo.Om0,
    #         cosmo.Ode0,
    #         cosmo.Tcmb0,
    #         cosmo.Neff,
    #         cosmo.Ob0,
    #     )
    # )
    plt.suptitle(
        "%s : H0=%s, Om0=%0.3f, Ode0=%0.3f, Tcmb0=%s, Neff=%0.2f, Ob0=%s"
        % (
            cosmology,
            cosmo.H0,
            cosmo.Om0,
            cosmo.Ode0,
            cosmo.Tcmb0,
            cosmo.Neff,
            cosmo.Ob0,
        ),
        y=1,
    )
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.15)
    plt.show()

    # for key in info.keys():
        #verboseprint("%s : %s" % (key, info[key]))
    return

cosmology_calculator()