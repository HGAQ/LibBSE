"""Produce a strict integration summary and JUnit without promoting historical baselines."""
from pathlib import Path
import argparse,json,xml.etree.ElementTree as ET
from compare_outputs import compare_directories,read_eigenpairs,compare_eigenpairs

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('work',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    report_root=args.output or args.work
    report_root.mkdir(parents=True,exist_ok=True)
    case_catalog=json.loads((Path(__file__).parent/'cases.json').read_text())['cases']
    report_records=[]
    junit_root=ET.Element('testsuite',name='LibBSE integration regression',tests=str(len(case_catalog)))
    report_lines=['# LibBSE integration regression results','',
        'Frozen reference and candidate use the same physical source snapshot in this initial campaign. PASS establishes a reproducible migration baseline; it is not independent physical validation.', '',
        '| Case | Status | Max eigenvalue difference / eV | Amplitude subspace error | Max relative W difference | Historical eigenvalue difference / eV | Historical amplitude subspace error | Historical status |',
        '|---|---|---:|---:|---:|---:|---:|---|']
    for case_definition in case_catalog:
        case_name=case_definition['name']
        case_root=args.work/case_name
        junit_case=ET.SubElement(junit_root,'testcase',name=case_name,classname='LibBSE.integration')
        try:
            stored_comparison=json.loads((case_root/'comparison.json').read_text())
            require_screening='chi0' in (case_root/'candidate/libbse.in').read_text()
            comparison_metrics=compare_directories(case_root/'reference/libbse.d',case_root/'candidate/libbse.d',require_screening)
            if (case_root/'replay').exists():
                candidate_energies,candidate_amplitudes=read_eigenpairs(case_root/'candidate/libbse.d')
                replay_energies,replay_amplitudes=read_eigenpairs(case_root/'replay/libbse.d')
                comparison_metrics['W_replay_eigenpairs']=compare_eigenpairs(candidate_energies,candidate_amplitudes,replay_energies,replay_amplitudes)
                comparison_metrics['pass'] &= comparison_metrics['W_replay_eigenpairs']['pass']
            comparison_metrics['pass'] &= stored_comparison['inputs_unchanged']
            comparison_metrics['historical_template']=stored_comparison.get('historical_template')
            case_status='PASS' if comparison_metrics['pass'] else 'FAIL'
            if not comparison_metrics['pass']: ET.SubElement(junit_case,'failure',message='Numerical regression')
            maximum_w_error=max((item['relative_W_error'] for item in comparison_metrics['screening']['matrices']),default=0.)
            historical_error=stored_comparison.get('historical_template',{}).get('max_energy_error_eV')
            historical_label='not applicable' if historical_error is None else f'{historical_error:.6g}'
            historical_metrics=stored_comparison.get('historical_template') or {}
            historical_amplitude=historical_metrics.get('max_subspace_sine')
            historical_amplitude_label='—' if historical_amplitude is None else f'{historical_amplitude:.6g}'
            historical_status='not applicable' if 'pass' not in historical_metrics else ('PASS' if historical_metrics['pass'] else 'DIFF')
            w_label=f'{maximum_w_error:.6g}' if require_screening else 'not reconstructed'
            report_lines.append(f'| {case_name} | {case_status} | {comparison_metrics["eigenpairs"]["max_energy_error_eV"]:.6g} | {comparison_metrics["eigenpairs"]["max_subspace_sine"]:.6g} | {w_label} | {historical_label} | {historical_amplitude_label} | {historical_status} |')
            report_records.append({'case':case_name,'status':case_status,**comparison_metrics})
        except (OSError,ValueError,KeyError) as error:
            ET.SubElement(junit_case,'failure',message=str(error))
            report_records.append({'case':case_name,'status':'INCOMPLETE_OR_INVALID','error':str(error)})
            report_lines.append(f'| {case_name} | INCOMPLETE_OR_INVALID | — | — | — | — | — | — |')
    failures=sum(item['status']!='PASS' for item in report_records)
    junit_root.set('failures',str(failures))
    ET.ElementTree(junit_root).write(report_root/'integration.xml',encoding='utf-8',xml_declaration=True)
    (report_root/'integration_summary.json').write_text(json.dumps(report_records,indent=2)+'\n')
    report_lines+=['','W metrics are populated for chi0 cases. Fixed-W cases test BSE consumption of identical input W; that path does not independently reconstruct W.', '',
                   'Historical differences are reported separately and do not automatically fail the migration comparison. AIMS and ABACUS are independent producers, not interchangeable numerical references.']
    (report_root/'INTEGRATION_RESULTS.md').write_text('\n'.join(report_lines)+'\n')
    print(f'{len(case_catalog)-failures}/{len(case_catalog)} integration cases passed')
    return int(failures>0)

if __name__=='__main__':
    raise SystemExit(main())
