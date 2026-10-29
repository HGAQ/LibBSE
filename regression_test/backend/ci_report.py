"""Render CTest JUnit as a GitHub summary without hiding failures or missing tests."""
from pathlib import Path
import argparse
import html
import os
import re
import tomllib
import xml.etree.ElementTree as ET

GENERAL_TESTS = [
    'reject_command_line', 'velocity_gauge', 'parameter', 'profiler',
    'memory_1', 'memory_2', 'memory_4', 'spectrum_mpi',
    'chi0_screening_serial', 'chi0_screening_mpi', 'uniform_qmesh',
    'aims_screened_serial', 'aims_screened_mpi', 'bse_files',
    'aims_w_converter', 'aims_mommat_converter', 'molecular_lri_comm_mpi',
    'elpa_solver_serial', 'elpa_solver_mpi', 'block_cyclic_optics_1',
    'block_cyclic_optics_2', 'block_cyclic_optics_3', 'block_cyclic_optics_4',
    'regression_comparator',
]


def failure_reason(test):
    sections = []
    for tag in ('failure', 'error', 'system-out', 'system-err'):
        for element in test.findall(tag):
            sections.extend([element.get('message', ''), element.text or ''])
    details = '\n'.join(sections).strip()
    relevant = [line.strip() for line in details.splitlines()
                if re.search(r'error|exception|failed|mismatch|False|timeout|not found', line, re.I)]
    reason = ' / '.join(relevant[-4:]) or details[-600:] or 'CTest reported failure; see the test log.'
    return reason[:900], details[-10000:]


def suite_report(path, title, expected_names, outcome):
    records = {}
    report_error = ''
    if path.exists():
        try:
            for test in ET.parse(path).getroot().iter('testcase'):
                name = test.get('name', '(unnamed)')
                if name in records:
                    raise ValueError(f'Duplicate test: {name}')
                records[name] = test
        except (ET.ParseError, ValueError) as error:
            records = {}
            report_error = str(error)
    else:
        report_error = 'JUnit report missing'
    names = list(expected_names) + sorted(set(records) - set(expected_names))
    passed = 0
    rows = []
    failure_details = []
    for name in names:
        test = records.get(name)
        if test is None:
            status = '🔴 NOT RUN'
            reason = report_error or 'Expected test missing from JUnit'
            reason += f'; stage: {outcome}'
        elif test.find('failure') is not None or test.find('error') is not None:
            status = '🔴 FAIL'
            reason, details = failure_reason(test)
            failure_details.append((name, details))
        elif test.find('skipped') is not None or test.get('status') in ('notrun', 'disabled'):
            status = '🔴 SKIPPED'
            reason = 'Test was skipped; this does not count as passed.'
        else:
            status = '🟢 PASS'
            reason = '—'
            passed += 1
        safe_reason = html.escape(reason).replace('|', '&#124;').replace('\n', '<br>')
        rows.append(f'| {html.escape(name)} | {status} | {safe_reason} |')
    complete = passed == len(names) and not report_error and outcome == 'success'
    marker = '🟢' if complete else '🔴'
    lines = [f'## {marker} {title}: {passed}/{len(names)}', '',
             f'Stage status: **{outcome}**.', '',
             '| Test | Result | Failure reason |', '|---|---|---|', *rows, '']
    if report_error:
        lines += [f'Report error: {html.escape(report_error)}', '']
    for name, details in failure_details:
        lines += ['<details>', f'<summary>{html.escape(name)} — failure output</summary>', '',
                  '<pre>' + html.escape(details) + '</pre>', '', '</details>', '']
    return lines, complete


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', type=Path, required=True)
    args = parser.parse_args()
    case_path = Path(__file__).resolve().parents[1] / 'physical_testcases/si222/cases.toml'
    physical_names = ['LibBSE_real_Si222_' + name for name in tomllib.loads(case_path.read_text())]
    compile_outcome = os.environ.get('COMPILE_OUTCOME', 'unknown')
    marker = '🟢' if compile_outcome == 'success' else '🔴'
    lines = ['# LibBSE test report', '', f'{marker} **Compile LibBSE: {compile_outcome}**', '']
    complete = compile_outcome == 'success'
    for filename, title, names, stage in [
        ('regression.xml', 'General Cases', ['LibBSE_' + name for name in GENERAL_TESTS], 'GENERAL_OUTCOME'),
        ('real-si.xml', 'Physical Cases', physical_names, 'PHYSICAL_OUTCOME'),
    ]:
        suite_lines, suite_complete = suite_report(args.build / filename, title, names,
                                                  os.environ.get(stage, 'unknown'))
        lines.extend(suite_lines)
        complete &= suite_complete
    if compile_outcome != 'success':
        for path in [args.build / 'configure.log', args.build / 'build.log',
                     Path('ci-deps/elpa-build/config.log')]:
            if path.exists():
                tail = '\n'.join(path.read_text(errors='replace').splitlines()[-50:])
                lines += [f'### {path.name}', '<pre>' + html.escape(tail[-10000:]) + '</pre>', '']
        lines += ['See the failed dependency/build step for the original error.', '']
    report = '\n'.join(lines) + '\n'
    args.build.mkdir(parents=True, exist_ok=True)
    (args.build / 'report.md').write_text(report)
    summary = os.environ.get('GITHUB_STEP_SUMMARY')
    if summary:
        with Path(summary).open('a') as handle:
            handle.write(report)
    print(report)
    return 0 if complete else 1


if __name__ == '__main__':
    raise SystemExit(main())
