module type has_iq = {
  type dtype
  val Iq: (q: dtype) -> (local_value: []dtype) -> dtype
  val Iqxy: (qx: dtype) -> (qy: dtype) -> (local_value: []dtype) -> dtype
  val form_volume: (local_value: []dtype) -> dtype
}

module for_iq (for_iq: (real: real) -> (has_iq with dtype = real.t))
              (real: real) = {
  module Iq = for_iq real
  type dtype = Iq.dtype

  let run_kernel(num_pars: i32, num_active: i32, nq : i32, call_details_num_evals: i32, call_details_buffer : []i32,
                 values : []dtype, q_input : []dtype, cutoff : dtype): []dtype =

    let background = values[1]
    let parameters = values[ 2: num_pars+2 ]
    let scatterings = map (\q -> Iq.Iq q parameters) q_input
    in if (num_active == 0)
       then
    let pd_norm = Iq.form_volume parameters
        let scale = real.(if pd_norm == (from_i32 0) then values[0] else values[0]/pd_norm)
        let res = map (\(r:dtype) : dtype ->
                       let out = real.(scale*r + background)
                       in out
                      ) scatterings
        in res[0:nq]
      else

    let scale = values[0]
    let spherical_correction = real.from_i32(1)
    let weight0 = real.from_i32(1)

    let zeroes = replicate nq (real.from_i32(0))
    let inds = iota nq
    let initial_result = zip zeroes inds

    let step = i32.(max (1000000 / (nq + 1)) 1)
    let pd_norm = real.from_i32(0)

    let res' = unsafe map (\r sc ->
                           loop r for pd_norm in [0] do
                           if real.(from_i32(1) > cutoff)
                           then real.(r + sc)
                           else r
                          ) zeroes scatterings
    let scale = if 2*step < call_details_num_evals then values[0] real./ real.from_i32 (call_details_num_evals - step) else values[0]
    let background = values[1]
    let output = map (\(r:dtype) : dtype ->
                      let out = real.(scale*r + background)
                      in out
                     ) res'
    in output

  let run_kernel_2d(num_pars: i32, num_active: i32, nq : i32, call_details_num_evals: i32, call_details_buffer : []i32,
                   values : []dtype, qx_input : []dtype, qy_input : []dtype, cutoff : dtype): []dtype =

    let background = values[1]
    let parameters = values[2:num_pars+2]
    let scatterings = map (\qx qy -> Iq.Iqxy qx qy parameters) qx_input qy_input
    in if (num_active == 0)
       then
    let pd_norm = Iq.form_volume parameters
    let scale = real.(if pd_norm == (from_i32 0) then values[0] else values[0]/pd_norm)
    let res = map (\(r:dtype) : dtype ->
                   let out = real.(scale*r + background)
                   in out
                  ) scatterings
    in res[0:nq]
       else
    let scale = values[0]
    let pd_value = values[2+num_pars : 2+num_pars]
    let weight0 = real.from_i32(1)
    
    let zeroes = replicate nq (real.from_i32(0))
    let inds = iota nq
    let initial_result = zip zeroes inds
    
    let step = i32.(max (1000000 / (nq + 1)) 1)
    let pd_norm = real.from_i32(0)
    
    let res' = unsafe map (\r sc ->
                           loop r for pd_norm in [0] do
                           if real.(from_i32(1) > cutoff)
                           then real.(r + sc)
                           else r
                          ) zeroes scatterings
    let scale = if 2*step < call_details_num_evals then values[0] real./ real.from_i32 (call_details_num_evals - step) else values[0]
    let background = values[1]
    let output = map (\(r:dtype) : dtype ->
                      let out = real.(scale*r + background)
                      in out
                     ) res'
    in output
}