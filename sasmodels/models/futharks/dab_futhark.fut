import "header/for_iq"

module dab (real: real) = {
  type dtype = real.t

  let Iq (q : dtype) (local_values : []dtype): dtype =
    let cor_length = local_values[0]
    let numerator = cor_length real.** real.from_i32(3)
    let denominator = (real.from_i32(1) real.+ ((q real.* cor_length) real.**
                                                real.from_i32(2))
                      ) real.** real.from_i32(2)
    in numerator real./ denominator

  -- using default Iqxy definition
  let Iqxy (qx: dtype) (qy: dtype) (local_value: []dtype): dtype =
    let two = real.from_i32(2)
    let q = real.(sqrt (qx**two + qy**two))
    in Iq q local_value

  -- using default form_volume
  let form_volume (local_value : []dtype) : dtype =
    real.from_i32(1)
}

module kernel_float32 = for_iq dab f32
module kernel_float64 = for_iq dab f64

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
