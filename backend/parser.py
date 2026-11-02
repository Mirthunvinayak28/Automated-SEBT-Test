"""
backend/parser.py
══════════════════════════════════════════════════════════════════════
Parses SEBT pipeline outputs:
  - Detailed movement records CSV (*_depth_results.csv)
  - Structured summary CSV (*_sebt_summary.csv)
══════════════════════════════════════════════════════════════════════
"""
import os
import csv
from typing import Dict, Any, List, Optional

ALL_SEBT_DIRECTIONS = ["ANT", "AM", "M", "PM", "POST", "PL", "LAT", "AL"]

def _safe_float(val: Any) -> Optional[float]:
    if val is None or val == "" or str(val).strip() in ("N/A", "None", "--", "UNAVAILABLE"):
        return None
    try:
        s = str(val).strip().rstrip('%')
        return float(s)
    except (ValueError, TypeError):
        return None

def _safe_int(val: Any) -> Optional[int]:
    if val is None or val == "" or str(val).strip() in ("N/A", "None", "--"):
        return None
    try:
        return int(float(str(val).strip()))
    except (ValueError, TypeError):
        return None

def parse_depth_results_csv(csv_path: str) -> List[Dict[str, Any]]:
    """
    Parses the detailed trial-by-trial movement records CSV.
    Preserves all 36 fields produced by the pipeline.
    """
    if not os.path.exists(csv_path):
        return []

    movements = []
    with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            start_t = _safe_float(row.get('start_time_sec'))
            max_t = _safe_float(row.get('max_time_sec'))
            duration = round(max_t - start_t, 2) if (start_t is not None and max_t is not None) else None

            rec = {
                'movement': _safe_int(row.get('movement')),
                'moving_ankle': row.get('moving_ankle', '').strip(),
                'start_frame': _safe_int(row.get('start_frame')),
                'max_frame': _safe_int(row.get('max_frame')),
                'start_time_sec': start_t,
                'max_time_sec': max_t,
                'duration_sec': duration,
                'initial_x': _safe_int(row.get('initial_x')),
                'initial_y': _safe_int(row.get('initial_y')),
                'max_x': _safe_int(row.get('max_x')),
                'max_y': _safe_int(row.get('max_y')),
                'pixel_distance_px': _safe_float(row.get('pixel_distance_px')),
                'initial_depth_m': _safe_float(row.get('initial_depth_m')),
                'max_depth_m': _safe_float(row.get('max_depth_m')),
                'depth_difference_m': _safe_float(row.get('depth_difference_m')),
                'initial_X_m': _safe_float(row.get('initial_X_m')),
                'initial_Y_m': _safe_float(row.get('initial_Y_m')),
                'initial_Z_m': _safe_float(row.get('initial_Z_m')),
                'max_X_m': _safe_float(row.get('max_X_m')),
                'max_Y_m': _safe_float(row.get('max_Y_m')),
                'max_Z_m': _safe_float(row.get('max_Z_m')),
                'distance_3d_cm': _safe_float(row.get('distance_3d_cm')),
                'stance_leg': row.get('stance_leg', '').strip(),
                'detected_direction': row.get('detected_direction', '').strip(),
                'direction_confidence': row.get('direction_confidence', '').strip(),
                'direction_confirmed_frame': _safe_int(row.get('direction_confirmed_frame')),
                'direction_confirmed_time': _safe_float(row.get('direction_confirmed_time')),
                'directional_displacement_cm': _safe_float(row.get('directional_displacement_cm')),
                'manual_stance_leg_length_cm': _safe_float(row.get('manual_stance_leg_length_cm')),
                'video_estimated_stance_leg_length_cm': _safe_float(row.get('video_estimated_stance_leg_length_cm')),
                'manual_leg_length_abs_error_cm': _safe_float(row.get('manual_leg_length_abs_error_cm')),
                'manual_leg_length_pct_error': _safe_float(row.get('manual_leg_length_pct_error')),
                'valid_directional_reach_distance_cm': _safe_float(row.get('valid_directional_reach_distance_cm')),
                'normalized_score_manual_pct': _safe_float(row.get('normalized_score_manual_pct')),
                'normalized_score_video_pct': _safe_float(row.get('normalized_score_video_pct')),
                'scoring_validity': row.get('scoring_validity', '').strip(),
                'scoring_exclusion_reason': row.get('scoring_exclusion_reason', '').strip(),
            }
            movements.append(rec)
    return movements

def parse_sebt_summary_csv(csv_path: str) -> Dict[str, Any]:
    """
    Parses the 4-section SEBT summary CSV.
    """
    if not os.path.exists(csv_path):
        return {
            'leg_lengths': {},
            'composite_scores': {},
            'limitations': []
        }

    sections: Dict[str, List[List[str]]] = {
        'leg_lengths': [],
        'movements': [],
        'composites': [],
        'limitations': []
    }

    current_section = None
    with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or not any(row):
                continue
            first_col = row[0].strip()
            if "SECTION 1" in first_col:
                current_section = 'leg_lengths'
                continue
            elif "SECTION 2" in first_col:
                current_section = 'movements'
                continue
            elif "SECTION 3" in first_col:
                current_section = 'composites'
                continue
            elif "SECTION 4" in first_col:
                current_section = 'limitations'
                continue

            if current_section:
                sections[current_section].append([c.strip() for c in row])

    # Parse Section 1: Leg Lengths
    leg_lengths = {}
    if len(sections['leg_lengths']) > 1:
        # Header is row 0
        for row in sections['leg_lengths'][1:]:
            if len(row) >= 5:
                leg_name = row[0].upper()
                leg_lengths[leg_name] = {
                    'leg': leg_name,
                    'manual_cm': _safe_float(row[1]),
                    'video_est_cm': _safe_float(row[2]),
                    'abs_error_cm': _safe_float(row[3]),
                    'pct_error': _safe_float(row[4]),
                    'note': row[5] if len(row) > 5 else ''
                }

    # Parse Section 3: Composites
    composite_scores = {}
    if len(sections['composites']) > 1:
        for row in sections['composites'][1:]:
            if len(row) >= 7:
                grp_key = row[0].strip()
                status_lbl = row[1].strip()
                denom = _safe_int(row[2]) or 0
                inc_dirs_str = row[3].strip()
                inc_dirs = [d.strip() for d in inc_dirs_str.split(';') if d.strip() and d.strip() != 'None']
                valid_cnt = _safe_int(row[4]) or 0
                m_score = _safe_float(row[5])
                v_score = _safe_float(row[6])

                missing_dirs = [d for d in ALL_SEBT_DIRECTIONS if d not in inc_dirs]

                composite_scores[grp_key] = {
                    'group': grp_key,
                    'status_label': status_lbl,
                    'denominator': denom,
                    'included_dirs': inc_dirs,
                    'missing_dirs': missing_dirs,
                    'valid_movements_count': valid_cnt,
                    'manual_composite_pct': m_score,
                    'video_composite_pct': v_score
                }

    # Parse Section 4: Limitations
    limitations = []
    if len(sections['limitations']) > 1:
        for row in sections['limitations'][1:]:
            if len(row) >= 2:
                limitations.append({
                    'item': row[0],
                    'description': row[1]
                })

    return {
        'leg_lengths': leg_lengths,
        'composite_scores': composite_scores,
        'limitations': limitations
    }

def get_complete_sebt_results(results_csv_path: str, summary_csv_path: str) -> Dict[str, Any]:
    """
    Compiles full structured results from both CSV files,
    including best trials by direction and observations.
    """
    movements = parse_depth_results_csv(results_csv_path)
    summary = parse_sebt_summary_csv(summary_csv_path)

    # Compute best trials per direction for stance groups
    groups = {'LEFT_STANCE': [], 'RIGHT_STANCE': [], 'COMBINED_ALL_STANCE': movements}
    for m in movements:
        if m.get('stance_leg') == 'LEFT':
            groups['LEFT_STANCE'].append(m)
        elif m.get('stance_leg') == 'RIGHT':
            groups['RIGHT_STANCE'].append(m)

    for grp_key, grp_records in groups.items():
        valid_recs = [r for r in grp_records if r.get('scoring_validity') == 'VALID']
        best_trials: Dict[str, Dict[str, Any]] = {}
        for r in valid_recs:
            d = r.get('detected_direction')
            dist = r.get('valid_reach_dist_cm')
            if d and dist is not None:
                if d not in best_trials or (best_trials[d].get('valid_reach_dist_cm') or 0.0) < dist:
                    best_trials[d] = {
                        'movement': r.get('movement'),
                        'direction': d,
                        'valid_reach_dist_cm': dist,
                        'norm_score_manual_pct': r.get('normalized_score_manual_pct'),
                        'norm_score_video_pct': r.get('normalized_score_video_pct'),
                        'moving_ankle': r.get('moving_ankle'),
                        'stance_leg': r.get('stance_leg')
                    }
        if grp_key in summary['composite_scores']:
            summary['composite_scores'][grp_key]['best_trials'] = best_trials

    # Global observations
    completed_count = len(movements)
    valid_count = sum(1 for m in movements if m.get('scoring_validity') == 'VALID')
    excluded_count = completed_count - valid_count
    detected_directions = sorted(list(set(m.get('detected_direction') for m in movements if m.get('detected_direction'))))
    missing_directions = [d for d in ALL_SEBT_DIRECTIONS if d not in detected_directions]

    left_leg = summary['leg_lengths'].get('LEFT', {})
    right_leg = summary['leg_lengths'].get('RIGHT', {})

    observations = {
        'total_movements': completed_count,
        'valid_movements': valid_count,
        'excluded_movements': excluded_count,
        'detected_directions': detected_directions,
        'missing_directions': missing_directions,
        'manual_left_leg_cm': left_leg.get('manual_cm'),
        'manual_right_leg_cm': right_leg.get('manual_cm'),
        'est_left_leg_cm': left_leg.get('video_est_cm'),
        'est_right_leg_cm': right_leg.get('video_est_cm'),
        'left_error_cm': left_leg.get('abs_error_cm'),
        'left_error_pct': left_leg.get('pct_error'),
        'right_error_cm': right_leg.get('abs_error_cm'),
        'right_error_pct': right_leg.get('pct_error'),
    }

    return {
        'movements': movements,
        'summary': summary,
        'observations': observations
    }
