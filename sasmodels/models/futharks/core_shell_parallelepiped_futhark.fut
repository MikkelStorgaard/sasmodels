import "header/for_iq"
import "lib/gauss76"
import "lib/sasmodel_consts"

module core_shell_parallelepiped (real: real) = {
  type dtype = real.t

  let gauss76Zreal = map real.from_f64 Gauss76.Gauss76Z
  let gauss76Wtreal = map real.from_f64 Gauss76.Gauss76Wt
  let zippedGauss = zip gauss76Zreal gauss76Wtreal
  let M_PI_2 = real.from_f64 sasmodel_consts.M_PI_2
  let M_PI_180 = real.from_f64 sasmodel_consts.M_PI_180

  let sas_sinx_x (x: dtype) : dtype =
    real.(if x == from_i32(0)
            then from_i32(1)
            else (sin(x) / x))

  let SINCOS (ang : dtype) : (dtype, dtype) = real.((sin ang, cos ang))

  let Iq (q : dtype) (local_values : []dtype): dtype =
    let core_sld = local_values[0]
    let arim_sld = local_values[1]
    let brim_sld = local_values[2]
    let crim_sld = local_values[3]
    let solvent_sld = local_values[4]
    let length_a = local_values[5]
    let length_b = local_values[6]
    let length_c = local_values[7]
    let thick_rim_a = local_values[8]
    let thick_rim_b = local_values[9]
    let thick_rim_c = local_values[10]

    let mu = real.(from_f64 0.5 * q * length_b)
    let a_scaled = real.( length_a / length_b )
    let c_scaled = real.( length_c / length_b )
    let ta = real.((a_scaled + from_i32 2 *thick_rim_a)/length_b)
    let tb = real.((a_scaled + from_i32 2 *thick_rim_b)/length_b)

    let Vin = real.(length_a * length_b * length_c)
    let V1 = real.(from_i32 2 * thick_rim_a * length_b * length_c)
    let V2 = real.(from_i32(2) * length_a * thick_rim_b * length_c)

    let drho0 = real.(core_sld-solvent_sld)
    let drhoA = real.(arim_sld-solvent_sld)
    let drhoB = real.(brim_sld-solvent_sld)

    let scale23 = real.(drhoA*V1)
    let scale14 = real.(drhoB*V2)
    let scale12 = real.(drho0*Vin - scale23 - scale14)

    let outer_totals = map (\(gaussZ,gaussWt) ->

      let sigma = real.(from_f64(0.5) * ( gaussZ + from_i32(1) ))
      let mu_proj = real.(mu * sqrt(from_i32(1) - sigma** from_i32(2)))
      let inner_totals = map (\(gaussZ, gaussWt) ->

        let uu = real.(from_f64(0.5) * (gaussZ + from_i32(1)))
        let (sin_uu, cos_uu) = real.(SINCOS(M_PI_2*uu))
        let si1 = real.(sas_sinx_x(mu_proj * sin_uu * a_scaled))
        let si2 = real.(sas_sinx_x(mu_proj * cos_uu))
        let si3 = real.(sas_sinx_x(mu_proj * sin_uu * ta))
        let si4 = real.(sas_sinx_x(mu_proj * cos_uu * tb))
        let form = real.(scale12*si1*si2 + scale23*si2*si3 + scale14*si1*si4)
        in real.(gaussWt * form * form)
        ) zippedGauss
      
      let inner_total = real.((reduce (+) (from_i32 0) inner_totals) / from_i32 2)
      let si = real.(sas_sinx_x(mu * c_scaled * sigma))
      in real.(gaussWt * inner_total * si * si)
      ) zippedGauss

    let outer_total = real.((reduce (+) (from_i32 0) outer_totals) / from_i32 2 )
    in real.(from_f64(0.0001) * outer_total)

  -- using default Iqxy definition
  let Iqxy (qx: dtype) (qy: dtype) (local_values: []dtype): dtype =
    let core_sld = local_values[0]
    let arim_sld = local_values[1]
    let brim_sld = local_values[2]
    let crim_sld = local_values[3]
    let solvent_sld = local_values[4]
    let length_a = local_values[5]
    let length_b = local_values[6]
    let length_c = local_values[7]
    let thick_rim_a = local_values[8]
    let thick_rim_b = local_values[9]
    let thick_rim_c = local_values[10]
    let theta = local_values[11]
    let phi = local_values[12]
    let psi = local_values[13]

    let dr0 = real.(core_sld-solvent_sld)
    let drA = real.(arim_sld-solvent_sld)
    let drB = real.(brim_sld-solvent_sld)
    let drC = real.(crim_sld-solvent_sld)


    let q = real.(sqrt(qx*qx + qy*qy))
    let qxhat = real.(qx/q)
    let qyhat = real.(qy/q)
    let (sin_theta, cos_theta) = real.(SINCOS(theta*M_PI_180))
    let (sin_phi, cos_phi) = real.(SINCOS(phi*M_PI_180))
    let (sin_psi, cos_psi) = real.(SINCOS(psi*M_PI_180))

    let xhat = real.( qxhat* (negate sin_phi * sin_psi + cos_theta*cos_phi*cos_psi) 
                      + qyhat* ( cos_phi*sin_psi + cos_theta*sin_phi*cos_psi))

    let yhat = real.(qxhat*((negate sin_phi)*cos_psi - cos_theta*cos_phi*sin_psi) 
                     + qyhat*( cos_phi*cos_psi - cos_theta*sin_phi*sin_psi))
    let zhat = real.(qxhat*(negate sin_theta*cos_phi)
                     + qyhat*(negate sin_theta*sin_phi))

    let Vin = real.(length_a * length_b * length_c)
    let V1 = real.(from_i32 2 * thick_rim_a * length_b * length_c)
    let V2 = real.(from_i32 2 * length_a * thick_rim_b * length_c)
    let V3 = real.(from_i32 2 * length_a * length_b * thick_rim_c)

    let ta = real.(length_a + from_i32 2*thick_rim_a)
    let tb = real.(length_a + from_i32 2*thick_rim_b)
    let tc = real.(length_a + from_i32 2*thick_rim_c)

    let siA = real.(sas_sinx_x(from_f64 0.5 *q*length_a*xhat))
    let siB = real.(sas_sinx_x(from_f64 0.5 *q*length_b*yhat))
    let siC = real.(sas_sinx_x(from_f64 0.5 *q*length_c*zhat))
    let siAt = real.(sas_sinx_x(from_f64 0.5 *q*ta*xhat))
    let siBt = real.(sas_sinx_x(from_f64 0.5 *q*tb*yhat))
    let siCt = real.(sas_sinx_x(from_f64 0.5 *q*tc*zhat))


    let f = real.( dr0*siA*siB*siC*Vin
                   + drA*(siAt-siA)*siB*siC*V1
                   + drB*siA*(siBt-siB)*siC*V2
                   + drC*siA*siB*(siCt*siCt-siC)*V3)
    in real.(from_f64 0.0001 * f * f)




  -- using default form_volume
  let form_volume (local_values : []dtype) : dtype =
    let length_a = local_values[5]
    let length_b = local_values[6]
    let length_c = local_values[7]
    let thick_rim_a = local_values[8]
    let thick_rim_b = local_values[9]
    let thick_rim_c = local_values[10]
    in real.( length_a * length_b * length_c +
              from_i32 2 * thick_rim_a * length_b * length_c + 
              from_i32 2 * thick_rim_b * length_a * length_c +
              from_i32 2 * thick_rim_c * length_a * length_b
            )
}

module kernel_float64 = for_iq core_shell_parallelepiped f64

entry kernel_float64 (num_pars: i32, num_active: i32, nq : i32, call_details_num_evals : i32, call_details_buffer : []i32,
                      values : []f64, q_input : []f64, cutoff : f64): []f64 =
  kernel_float64.run_kernel(num_pars, num_active, nq, call_details_num_evals, call_details_buffer,
                            values, q_input, cutoff)

entry kernel_float64_2d (num_pars: i32, num_active: i32, nq : i32, call_details_num_evals : i32, call_details_buffer : []i32,
                         values : []f64, qx_input : []f64, qy_input : []f64, cutoff : f64): []f64 =
  kernel_float64.run_kernel_2d(num_pars, num_active, nq, call_details_num_evals, call_details_buffer,
                            values, qx_input, qy_input, cutoff)