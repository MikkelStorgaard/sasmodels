double form_volume(double radius_equat_minor, double radius_equat_major, double radius_polar);
double Iq(double q, double sld, double sld_solvent,
    double radius_equat_minor, double radius_equat_major, double radius_polar);
double Iqxy(double qx, double qy, double sld, double sld_solvent,
    double radius_equat_minor, double radius_equat_major, double radius_polar, double theta, double phi, double psi);

//#define INVALID(v) (v.radius_equat_minor > v.radius_equat_major || v.radius_equat_major > v.radius_polar)


double form_volume(double radius_equat_minor, double radius_equat_major, double radius_polar)
{
    return M_4PI_3*radius_equat_minor*radius_equat_major*radius_polar;
}

//TODO: Need to check this - it doesn't really corresspond to documentation
double Fq(double q,
        double acosx2,
        double bsinx2,
        double ysq,
        double c2)
{
    double t = q*sqrt(acosx2 + bsinx2*(1.0-ysq) + c2*ysq);
    double fq = sph_j1c(t);
    return fq;
}
double Iq(double q,
    double sld,
    double sld_solvent,
    double radius_equat_minor,
    double radius_equat_major,
    double radius_polar)
{
    double sn, cn;
    // translate a point in [-1,1] to a point in [0, 1]
    const double zm = 0.5;
    const double zb = 0.5;
    double outer = 0.0;
    for (int i=0;i<N_POINTS_76;i++) {
        //const double cos_alpha = (Gauss76Z[i]*(upper-lower) + upper + lower)/2;
        const double x = 0.5*(Gauss76Z[i] + 1.0);
        SINCOS(M_PI_2*x, sn, cn);
        const double acosx2 = radius_equat_minor*radius_equat_minor*cn*cn;
        const double bsinx2 = radius_equat_major*radius_equat_major*sn*sn;
        const double c2 = radius_polar*radius_polar;

        double inner = 0.0;
        for (int j=0;j<N_POINTS_76;j++) {
            const double ysq = square(Gauss76Z[j]*zm + zb);
            inner += Gauss76Wt[j] * square(Fq(q,acosx2,bsinx2,ysq,c2));
        }
        outer += Gauss76Wt[i] * 0.5 * inner;
    }
    // translate dx in [-1,1] to dx in [lower,upper]
    const double fqsq = outer*zm;
    const double s = (sld - sld_solvent) * form_volume(radius_equat_minor, radius_equat_major, radius_polar);
    return 1.0e-4 * s * s * fqsq;
}

double Iqxy(double qx, double qy,
    double sld,
    double sld_solvent,
    double radius_equat_minor,
    double radius_equat_major,
    double radius_polar,
    double theta,
    double phi,
    double psi)
{
    double q, calpha, cmu, cnu;
    ORIENT_ASYMMETRIC(qx, qy, theta, phi, psi, q, calpha, cmu, cnu);

    const double t = q*sqrt(radius_equat_minor*radius_equat_minor*cnu*cnu
                          + radius_equat_major*radius_equat_major*cmu*cmu
                          + radius_polar*radius_polar*calpha*calpha);
    const double fq = sph_j1c(t);
    const double s = (sld - sld_solvent) * form_volume(radius_equat_minor, radius_equat_major, radius_polar);

    return 1.0e-4 * square(s * fq);
}

