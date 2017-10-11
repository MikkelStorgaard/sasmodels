import "header/for_iq"

module broad_peak (real: real) = {
  type dtype = real.t

  let Iq (q: dtype) (local_value: []dtype): dtype =
    let porod_scale = local_value[0]
    let porod_exp = local_value[1]
    let lorentz_scale = local_value[2]
    let lorentz_length = local_value[3]
    let peak_pos = local_value[4]
    let lorentz_exp = local_value[5]

    let z = real.(abs(q-peak_pos) * lorentz_length)
    in real.(porod_scale / q ** porod_exp +
             lorentz_scale / (from_i32 1 + z ** lorentz_exp))

  -- using default Iqxy definition
  let Iqxy (qx: dtype) (qy: dtype) (local_value: []dtype): dtype =
    let two = real.from_i32(2)
    let q = real.(sqrt (qx**two + qy**two))
    in Iq q local_value

  -- using default form_volume
  let form_volume (local_value : []dtype) : dtype =
    real.from_i32(1)
}

module kernel_float32 = for_iq broad_peak f32
module kernel_float64 = for_iq broad_peak f64

entry kernel_float32 (num_pars: i32, num_active: i32, nq : i32, call_details_num_evals : i32, call_details_buffer : []i32,
                      values : []f32, q_input : []f32, cutoff : f32): []f32 =
  kernel_float32.run_kernel(num_pars, num_active, nq, call_details_num_evals, call_details_buffer,
                            values, q_input, cutoff)


entry kernel_float64 (num_pars: i32, num_active: i32, nq : i32, call_details_num_evals : i32, call_details_buffer : []i32,
                      values : []f64, q_input : []f64, cutoff : f64): []f64 =
  kernel_float64.run_kernel(num_pars, num_active, nq, call_details_num_evals, call_details_buffer,
                            values, q_input, cutoff)

entry kernel_float32_2d (num_pars: i32, num_active: i32, nq : i32, call_details_num_evals : i32, call_details_buffer : []i32,
                         values : []f32, qx_input : []f32, qy_input : []f32, cutoff : f32): []f32 =
  kernel_float32.run_kernel_2d(num_pars, num_active, nq, call_details_num_evals, call_details_buffer,
                            values, qx_input, qy_input, cutoff)


entry kernel_float64_2d (num_pars: i32, num_active: i32, nq : i32, call_details_num_evals : i32, call_details_buffer : []i32,
                         values : []f64, qx_input : []f64, qy_input : []f64, cutoff : f64): []f64 =
  kernel_float64.run_kernel_2d(num_pars, num_active, nq, call_details_num_evals, call_details_buffer,
                            values, qx_input, qy_input, cutoff)