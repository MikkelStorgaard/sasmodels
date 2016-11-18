double form_volume(double radius, double length);
double Fq(double q, double sn, double cn,double radius, double length);
double orient_avg_1D(double q, double radius, double length);
double Iq(double q, double sld, double solvent_sld, double radius, double length);
double Iqxy(double qx, double qy, double sld, double solvent_sld,
    double radius, double length, double theta, double phi);

#define INVALID(v) (v.radius<0 || v.length<0)

double form_volume(double radius, double length)
{
    return M_PI*radius*radius*length;
}

double Fq(double q, double radius, double length, double sn, double cn)
{
    // precompute qr and qh to save time in the loop
    const double qr = q*radius;
    const double qh = q*0.5*length; 
    return sas_J1c(qr*sn) * sinc(qh*cn);
}

double orient_avg_1D(double q, double radius, double length)
{
    // translate a point in [-1,1] to a point in [0, pi/2]
    const double zm = M_PI_4;
    const double zb = M_PI_4; 

    double total = 0.0;
    for (int i=0; i<N_POINTS_76 ;i++) {
        const double alpha = Gauss76Z[i]*zm + zb;
        double sn, cn; // slots to hold sincos function output
        // alpha(theta,phi) the projection of the cylinder on the detector plane
        SINCOS(alpha, sn, cn);
        total += Gauss76Wt[i] * square( Fq(q, radius, length, sn, cn) ) * sn;
    }
    // translate dx in [-1,1] to dx in [lower,upper]
    return total*zm;
}

double Iq(double q,
    double sld,
    double solvent_sld,
    double radius,
    double length)
{
    const double s = (sld - solvent_sld) * form_volume(radius, length);
    return 1.0e-4 * s * s * orient_avg_1D(q, radius, length);
}


double Iqxy(double qx, double qy,
    double sld,
    double solvent_sld,
    double radius,
    double length,
    double theta,
    double phi)
{
    double q, sin_alpha, cos_alpha;
    ORIENT_SYMMETRIC(qx, qy, theta, phi, q, sin_alpha, cos_alpha);
    //printf("sn: %g cn: %g\n", sin_alpha, cos_alpha);
    const double s = (sld-solvent_sld) * form_volume(radius, length);
    const double form = Fq(q, sin_alpha, cos_alpha, radius, length);
    return 1.0e-4 * square(s * form);
}
