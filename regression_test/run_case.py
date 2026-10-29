"""Run a fixed-input baseline/candidate comparison without modifying fixtures."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,time
import numpy as np
from compare_outputs import compare_directories,read_eigenpairs,compare_eigenpairs,read_screening

def file_digest(source_path):
    with source_path.open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()

def parse_input(input_path):
    input_options={}
    for input_line in input_path.read_text().splitlines():
        input_line=input_line.split('#',1)[0].strip()
        if input_line:
            input_key,input_value=input_line.split(None,1)
            input_options[input_key]=input_value
    return input_options

def input_manifest(input_options):
    source_manifest={}
    for input_key in ('input_dir','qp_data','screened_dir'):
        source_path=Path(input_options[input_key])
        if not source_path.exists(): raise FileNotFoundError(source_path)
        source_files=[source_path] if source_path.is_file() else sorted(source_path.rglob('*'))
        for source_file in source_files:
            if source_file.is_file():
                source_manifest[str(source_file)]=file_digest(source_file)
    return source_manifest

def run_bse(executable,case_root,input_options,ranks):
    case_root.mkdir(parents=True,exist_ok=False)
    (case_root/'libbse.in').write_text(''.join(f'{key} {value}\n' for key,value in input_options.items()))
    thread_count=int(os.environ.get('OMP_NUM_THREADS','1'))
    command=['mpirun','-np',str(ranks),'--map-by',f'slot:PE={thread_count}','--bind-to','core',str(executable)]
    run_start=time.monotonic()
    with (case_root/'libbse.out').open('w') as stdout,(case_root/'libbse.err').open('w') as stderr:
        subprocess.run(command,cwd=case_root,stdout=stdout,stderr=stderr,check=True,timeout=3600)
    if 'Time of Total LibBSE execution' not in (case_root/'libbse.out').read_text():
        raise RuntimeError('Missing successful LibBSE completion')
    (case_root/'execution.json').write_text(json.dumps({'executable_sha256':file_digest(executable),'ranks':ranks,'elapsed_seconds':time.monotonic()-run_start,'command':command},indent=2)+'\n')

def screening_to_cache(output_root,cache_root,plasma_energy):
    screening_records=read_screening(output_root)
    if not screening_records: raise ValueError('No W diagnostics to replay')
    coarse_n=round(len(screening_records)**(1/3))
    if coarse_n**3!=len(screening_records): raise ValueError('Replay fixtures must use cubic q grids')
    binding_energy=0.
    if plasma_energy>0:
        dynamic_text=(output_root/'dynamical_singlet.dat').read_text()
        for dynamic_line in dynamic_text.splitlines():
            if dynamic_line.startswith('# binding_used '): binding_energy=float(dynamic_line.split()[-1])
        if binding_energy<=0: raise ValueError('Missing positive dynamic binding energy')
    cache_root.mkdir()
    cache_format='full_w_binary' if all(abs(key[3])<1e-14 for key in screening_records) else 'fhi_aims_w'
    for q_number,(q_frequency,screening_pair) in enumerate(sorted(screening_records.items()),1):
        sqrt_cut,inverse_dielectric=screening_pair
        if plasma_energy>0:
            inverse_eigenvalues,inverse_eigenvectors=np.linalg.eigh(inverse_dielectric)
            if inverse_eigenvalues.min() < -1e-7 or inverse_eigenvalues.max()>1+1e-7:
                raise ValueError('Nonpassive inverse dielectric')
            inverse_eigenvalues=np.clip(inverse_eigenvalues,0,1)
            effective_eigenvalues=1-plasma_energy*(1-inverse_eigenvalues)/(plasma_energy+binding_energy*np.sqrt(1-inverse_eigenvalues))
            inverse_dielectric=(inverse_eigenvectors*effective_eigenvalues)@inverse_eigenvectors.conj().T
        screened_w=sqrt_cut@inverse_dielectric@sqrt_cut
        if cache_format=='full_w_binary':
            with (cache_root/f'full_w_q_{q_number:06d}.bin').open('wb') as handle:
                handle.write(b'LIBBSE_FULL_W_V1')
                np.array([len(screened_w),coarse_n,coarse_n,coarse_n],np.int32).tofile(handle)
                np.array(q_frequency,float).tofile(handle)
                np.asarray(screened_w,dtype=np.complex128,order='C').tofile(handle)
        else:
            # Preserve the actual nonzero imaginary frequency; never relabel it as zero.
            auxiliary_rows,auxiliary_columns=np.indices(screened_w.shape)
            screening_table=np.column_stack((np.ones(screened_w.size),np.full(screened_w.size,q_frequency[3]),auxiliary_rows.ravel()+1,auxiliary_columns.ravel()+1,screened_w.real.ravel(),screened_w.imag.ravel()))
            screening_header=(f'quantity: w\nn_basbas n_freq q_index: {len(screened_w)} 1 {q_number}\n'
                              f'q_fractional: {q_frequency[0]} {q_frequency[1]} {q_frequency[2]}\n'
                              'Original RI auxiliary basis; atomic units; indices start at 1.')
            np.savetxt(cache_root/f'periodic_gw_w_q_{q_number}_rank_0.dat',screening_table,
                       fmt=['%d','%.17e','%d','%d','%.17e','%.17e'],header=screening_header)
    return {'format':cache_format,'q_count':len(screening_records),'plasma_eV':plasma_energy,'binding_eV':binding_energy}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--fixtures',required=True,type=Path)
    parser.add_argument('--work',required=True,type=Path)
    parser.add_argument('--case',required=True)
    parser.add_argument('--resume',action='store_true',help='Reuse only completed runs with matching inputs and executable hashes')
    parser.add_argument('--reference-exe',required=True,type=Path)
    parser.add_argument('--candidate-exe',required=True,type=Path)
    parser.add_argument('--reference-ranks',type=int,default=1)
    parser.add_argument('--candidate-ranks',type=int,default=1)
    args=parser.parse_args()
    case_catalog=json.loads((Path(__file__).parent/'cases.json').read_text())['cases']
    case_definition=next(item for item in case_catalog if item['name']==args.case)
    fixture_root=(args.fixtures/case_definition['template']).resolve()
    case_work=(args.work/args.case).resolve()
    if case_work.exists() and not args.resume:
        case_work=case_work.with_name(case_work.name+'_'+str(time.time_ns()))
    case_work.mkdir(parents=True,exist_ok=args.resume)
    input_options=parse_input(fixture_root/'libbse.in')
    for input_key in ('input_dir','qp_data','screened_dir'):
        input_options[input_key]=str((fixture_root/input_options[input_key]).resolve())
    input_options.update(case_definition['overrides'])
    input_options.update({'output_dir':'libbse.d','out_bse_eigenvectors':'true','out_screening_matrices':'true'})
    before_manifest=input_manifest(input_options)
    if args.resume and json.loads((case_work/'input_manifest.json').read_text())!=before_manifest:
        raise ValueError('Input files changed since the interrupted run')
    (case_work/'input_manifest.json').write_text(json.dumps(before_manifest,indent=2)+'\n')
    (case_work/'case.json').write_text(json.dumps(case_definition,indent=2)+'\n')
    for run_branch,run_executable,run_ranks in [('reference',args.reference_exe,args.reference_ranks),('candidate',args.candidate_exe,args.candidate_ranks)]:
        run_root=case_work/run_branch
        if args.resume:
            if parse_input(run_root/'libbse.in')!=input_options:
                raise ValueError('Resumed physical input differs')
            execution_record=json.loads((run_root/'execution.json').read_text())
            if execution_record['executable_sha256']!=file_digest(run_executable):
                raise ValueError('Resumed executable differs')
            if 'Time of Total LibBSE execution' not in (run_root/'libbse.out').read_text():
                raise ValueError('Cannot reuse incomplete execution')
        else:
            run_bse(run_executable.resolve(),run_root,input_options,run_ranks)
    require_screening='chi0' in input_options['screened_format']
    comparison_metrics=compare_directories(case_work/'reference/libbse.d',case_work/'candidate/libbse.d',require_screening)
    # A historical result is a version-drift report, not an automatically approved oracle.
    if not case_definition['overrides']:
        old_energies,old_amplitudes=read_eigenpairs(fixture_root/'libbse.d')
        new_energies,new_amplitudes=read_eigenpairs(case_work/'candidate/libbse.d')
        comparison_metrics['historical_template']=compare_eigenpairs(old_energies,old_amplitudes,new_energies,new_amplitudes)
    if require_screening:
        if args.resume:
            for replay_name in ('replay','replay_W'):
                replay_root=case_work/replay_name
                if replay_root.exists(): replay_root.rename(case_work/(replay_name+'_previous_'+str(time.time_ns())))
        plasma_energy=float(input_options.get('bse_plasma_energy_ev','0'))
        comparison_metrics['replay_model']=screening_to_cache(case_work/'candidate/libbse.d',case_work/'replay_W',plasma_energy)
        replay_options=dict(input_options)
        replay_options.update({'screened_format':comparison_metrics['replay_model']['format'],'screened_dir':str(case_work/'replay_W'),'bse_plasma_energy_ev':'0','out_screening_matrices':'false'})
        run_bse(args.candidate_exe.resolve(),case_work/'replay',replay_options,args.candidate_ranks)
        candidate_energies,candidate_amplitudes=read_eigenpairs(case_work/'candidate/libbse.d')
        replay_energies,replay_amplitudes=read_eigenpairs(case_work/'replay/libbse.d')
        comparison_metrics['W_replay_eigenpairs']=compare_eigenpairs(candidate_energies,candidate_amplitudes,replay_energies,replay_amplitudes)
        comparison_metrics['pass'] &= comparison_metrics['W_replay_eigenpairs']['pass']
    comparison_metrics['inputs_unchanged']=before_manifest==input_manifest(input_options)
    comparison_metrics['pass'] &= comparison_metrics['inputs_unchanged']
    (case_work/'comparison.json').write_text(json.dumps(comparison_metrics,indent=2)+'\n')
    print(json.dumps({'case':args.case,'pass':comparison_metrics['pass'],'eigenpairs':comparison_metrics['eigenpairs']},indent=2))
    return 0 if comparison_metrics['pass'] else 1

if __name__=='__main__':
    raise SystemExit(main())
