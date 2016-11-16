double form_volume(double length_a, double b2a_ratio, double c2a_ratio, double thickness);
double Iq(double q, double sld, double solvent_sld, double length_a, 
          double b2a_ratio, double c2a_ratio, double thickness);

double form_volume(double length_a, double b2a_ratio, double c2a_ratio, double thickness)
{
    double length_b = length_a * b2a_ratio;
    double length_c = length_a * c2a_ratio;
    double a_core = length_a - 2.0*thickness;
    double b_core = length_b - 2.0*thickness;
    double c_core = length_c - 2.0*thickness;
    double vol_core = a_core * b_core * c_core;
    double vol_total = length_a * length_b * length_c;
    double vol_shell = vol_total - vol_core;
    return vol_shell;
}

double Fq(double q,
    double a_half,
    double b_half,
    double thickness,
    double phi,
    double sin_theta,
    double vol_core,
    double vol_total,
    double termC1,
    double termC2)
{
    double sin_phi, cos_phi;
    SINCOS(phi, sin_phi, cos_phi);
    // Amplitude AP from eqn. (13), rewritten to avoid round-off effects when arg=0
    const double termA1 = sinc(q * a_half * sin_theta * sin_phi);
    const double termA2 = sinc(q * (a_half-thickness) * sin_theta * sin_phi);

    const double termB1 = sinc(q * b_half * sin_theta * cos_phi);
    const double termB2 = sinc(q * (b_half-thickness) * sin_theta * cos_phi);

    const double AP1 = vol_total * termA1 * termB1 * termC1;
    const double AP2 = vol_core * termA2 * termB2 * termC2;

    double fq = AP1-AP2;

    return fq;
}
double Iq(double q,
    double sld,
    double solvent_sld,
    double length_a,
    double b2a_ratio,
    double c2a_ratio,
    double thickness)
{
    const double length_b = length_a * b2a_ratio;
    const double length_c = length_a * c2a_ratio;
    const double a_half = 0.5 * length_a;
    const double b_half = 0.5 * length_b;
    const double c_half = 0.5 * length_c;
    const double vol_total = length_a * length_b * length_c;
    const double vol_core = 8.0 * (a_half-thickness) * (b_half-thickness) * (c_half-thickness);

    //Integration limits to use in Gaussian quadrature
    const double v1a = 0.0;
    const double v1b = M_PI_2;  //theta integration limits
    const double v2a = 0.0;
    const double v2b = M_PI_2;  //phi integration limits
    
    double outer_sum = 0.0;
    for(int i=0; i<N_POINTS_76; i++) {

        const double theta = 0.5 * ( Gauss76Z[i]*(v1b-v1a) + v1a + v1b );
        double sin_theta, cos_theta;
        SINCOS(theta, sin_theta, cos_theta);

        const double termC1 = sinc(q * c_half * cos(theta));
        const double termC2 = sinc(q * (c_half-thickness)*cos(theta));

        double inner_sum = 0.0;
        for(int j=0; j<N_POINTS_76; j++) {

            const double phi = 0.5 * ( Gauss76Z[j]*(v2b-v2a) + v2a + v2b );

            inner_sum += Gauss76Wt[j] * square(Fq(q, a_half, b_half, thickness,
                            phi, sin_theta, vol_core, vol_total,
                            termC1, termC2));
        }
        inner_sum *= 0.5 * (v2b-v2a);

        outer_sum += Gauss76Wt[i] * inner_sum * sin(theta);
    }
    outer_sum *= 0.5*(v1b-v1a);

    // Normalize as in Eqn. (15) without the volume factor (as cancels with (V*DelRho)^2 normalization)
    // The factor 2 is due to the different theta integration limit (pi/2 instead of pi)
    const double form = outer_sum/M_PI_2;

    // Multiply by contrast^2. Factor corresponding to volume^2 cancels with previous normalization.
    const double delrho = sld - solvent_sld;

    // Convert from [1e-12 A-1] to [cm-1]
    return 1.0e-4 * delrho * delrho * form;
}
