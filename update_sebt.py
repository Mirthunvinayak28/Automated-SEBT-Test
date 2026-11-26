import sys

with open('sebt_video_depth_test.py', 'r') as f:
    content = f.read()

# 1. Update _landmark_px
old_landmark = '''def _landmark_px(landmark, w, h):
    return int(landmark.x * w), int(landmark.y * h)'''

new_landmark = '''def _landmark_px(landmark, w, h):
    return int(landmark.x * w), int(landmark.y * h)'''

# We don't change _landmark_px, we do validation in the main loop to handle raw coordinates.

# 2. Update initialization
old_init = '''        l_ankles_calib = []
        r_ankles_calib = []
        l_ankle_ref = None
        r_ankle_ref = None
        l_depth_ref = 0.0
        r_depth_ref = 0.0
        initial_frame_saved = False'''

new_init = '''        l_ankles_calib = []
        r_ankles_calib = []
        l_ankle_ref = None
        r_ankle_ref = None
        l_depth_ref = 0.0
        r_depth_ref = 0.0
        l_center_ref = 0.0
        r_center_ref = 0.0
        l_raw_ref = None
        r_raw_ref = None
        l_pos_smooth = None
        r_pos_smooth = None
        initial_frame_saved = False'''
content = content.replace(old_init, new_init)

# 3. Update max vars
old_max_vars = '''        max_frame_idx = 0
        max_timestamp = 0
        start_frame_idx = 0
        start_timestamp = 0'''

new_max_vars = '''        max_frame_idx = 0
        max_timestamp = 0
        start_frame_idx = 0
        start_timestamp = 0
        max_raw = None'''
content = content.replace(old_max_vars, new_max_vars)

# 4. Replace Pose reading block
old_pose = '''            if pose_result.pose_landmarks:
                pl = pose_result.pose_landmarks[0]
                draw_pose_skeleton(frame, pl, w, h)
                l_pos = _landmark_px(pl[L_ANKLE], w, h)
                r_pos = _landmark_px(pl[R_ANKLE], w, h)
                
                # Highlight ankles
                cv2.circle(frame, l_pos, 8, (255, 0, 0), -1) # Blue Left
                cv2.circle(frame, r_pos, 8, (0, 0, 255), -1) # Red Right
                draw_text(frame, 'L', (l_pos[0]+10, l_pos[1]), (255, 0, 0))
                draw_text(frame, 'R', (r_pos[0]+10, r_pos[1]), (0, 0, 255))'''

new_pose = '''            if pose_result.pose_landmarks:
                pl = pose_result.pose_landmarks[0]
                draw_pose_skeleton(frame, pl, w, h)
                
                l_px = _landmark_px(pl[L_ANKLE], w, h)
                r_px = _landmark_px(pl[R_ANKLE], w, h)
                
                if 0 <= l_px[0] < w and 0 <= l_px[1] < h:
                    l_raw_curr = (pl[L_ANKLE].x, pl[L_ANKLE].y)
                    if l_pos_smooth is None: l_pos_smooth = list(l_px)
                    else: l_pos_smooth = [0.6 * l_pos_smooth[0] + 0.4 * l_px[0], 0.6 * l_pos_smooth[1] + 0.4 * l_px[1]]
                    l_pos = (int(l_pos_smooth[0]), int(l_pos_smooth[1]))
                else:
                    l_pos = None
                    l_raw_curr = None
                    
                if 0 <= r_px[0] < w and 0 <= r_px[1] < h:
                    r_raw_curr = (pl[R_ANKLE].x, pl[R_ANKLE].y)
                    if r_pos_smooth is None: r_pos_smooth = list(r_px)
                    else: r_pos_smooth = [0.6 * r_pos_smooth[0] + 0.4 * r_px[0], 0.6 * r_pos_smooth[1] + 0.4 * r_px[1]]
                    r_pos = (int(r_pos_smooth[0]), int(r_pos_smooth[1]))
                else:
                    r_pos = None
                    r_raw_curr = None
                
                # Highlight ankles
                if l_pos:
                    cv2.circle(frame, l_pos, 8, (255, 0, 0), -1) # Blue Left
                    draw_text(frame, 'L', (l_pos[0]+10, l_pos[1]), (255, 0, 0))
                if r_pos:
                    cv2.circle(frame, r_pos, 8, (0, 0, 255), -1) # Red Right
                    draw_text(frame, 'R', (r_pos[0]+10, r_pos[1]), (0, 0, 255))'''
content = content.replace(old_pose, new_pose)


# 5. Calibration depth setup
old_calib_depth = '''                    # Run depth on ONE initial frame
                    depth_map = run_depth_inference(depth_model, frame, device)
                    _, l_depth_ref = sample_depth_neighborhood(depth_map, l_ankle_ref)
                    _, r_depth_ref = sample_depth_neighborhood(depth_map, r_ankle_ref)
                    
                    print(f"Calibration Complete: L={l_ankle_ref}, R={r_ankle_ref}")
                    print(f"Initial Depths: L={l_depth_ref:.4f}m, R={r_depth_ref:.4f}m")'''

new_calib_depth = '''                    # Run depth on ONE initial frame
                    depth_map = run_depth_inference(depth_model, frame, device)
                    l_center_ref, l_depth_ref = sample_depth_neighborhood(depth_map, l_ankle_ref)
                    r_center_ref, r_depth_ref = sample_depth_neighborhood(depth_map, r_ankle_ref)
                    
                    l_raw_ref = l_raw_curr
                    r_raw_ref = r_raw_curr
                    
                    print(f"Calibration Complete: L={l_ankle_ref}, R={r_ankle_ref}")
                    print(f"Initial Depths (median): L={l_depth_ref:.4f}m, R={r_depth_ref:.4f}m")'''
content = content.replace(old_calib_depth, new_calib_depth)


# 6. Max update tracking raw
old_max_save = '''                        max_pixel_dist = curr_dist
                        max_pos = curr_pos
                        max_frame_img = frame.copy()
                        max_frame_idx = frame_idx
                        max_timestamp = timestamp'''

new_max_save = '''                        max_pixel_dist = curr_dist
                        max_pos = curr_pos
                        max_frame_img = frame.copy()
                        max_frame_idx = frame_idx
                        max_timestamp = timestamp
                        max_raw = l_raw_curr if moving_ankle == 'LEFT' else r_raw_curr'''

content = content.replace(old_max_save, new_max_save)


# 7. Print for S_MOVEMENT
old_movement_print = '''                            # Run Metric Depth on exactly the max frame
                            depth_map = run_depth_inference(depth_model, max_frame_img, device)
                            _, max_depth = sample_depth_neighborhood(depth_map, max_pos)
                            
                            initial_depth = l_depth_ref if moving_ankle == 'LEFT' else r_depth_ref
                            
                            depth_diff = abs(max_depth - initial_depth)
                            
                            p_init_3d = pixel_to_3d(ref_pos, initial_depth, CX, CY, FX, FY)
                            p_max_3d = pixel_to_3d(max_pos, max_depth, CX, CY, FX, FY)
                            
                            dist_3d = calculate_3d_distance(p_init_3d, p_max_3d) * 100.0 # cm
                            
                            print("\\n" + "="*50)
                            print(f"MOVEMENT {curr_movement_num}")
                            print("="*50)
                            print(f"Moving ankle: {moving_ankle}")
                            print(f"Start frame: {start_frame_idx}")
                            print(f"Maximum frame: {max_frame_idx}")
                            print(f"Maximum timestamp: {max_timestamp:.2f} sec")
                            print(f"Initial pixel position:\\n{ref_pos}")
                            print(f"Maximum pixel position:\\n{max_pos}")
                            print(f"Pixel displacement:\\n{max_pixel_dist:.2f} px")
                            print(f"Initial metric depth:\\n{initial_depth:.4f} m")
                            print(f"Maximum metric depth:\\n{max_depth:.4f} m")
                            print(f"Depth-axis difference:\\n{depth_diff:.4f} m")
                            print(f"Initial 3D position:\\n({p_init_3d[0]:.4f}, {p_init_3d[1]:.4f}, {p_init_3d[2]:.4f}) m")
                            print(f"Maximum 3D position:\\n({p_max_3d[0]:.4f}, {p_max_3d[1]:.4f}, {p_max_3d[2]:.4f}) m")
                            print(f"3D displacement:\\n{dist_3d:.2f} cm")
                            print("="*50)'''

new_movement_print = '''                            # Run Metric Depth on exactly the max frame
                            depth_map = run_depth_inference(depth_model, max_frame_img, device)
                            max_center, max_depth = sample_depth_neighborhood(depth_map, max_pos)
                            
                            initial_depth = l_depth_ref if moving_ankle == 'LEFT' else r_depth_ref
                            initial_center = l_center_ref if moving_ankle == 'LEFT' else r_center_ref
                            ref_raw = l_raw_ref if moving_ankle == 'LEFT' else r_raw_ref
                            
                            depth_diff = abs(max_depth - initial_depth)
                            
                            p_init_3d = pixel_to_3d(ref_pos, initial_depth, CX, CY, FX, FY)
                            p_max_3d = pixel_to_3d(max_pos, max_depth, CX, CY, FX, FY)
                            
                            dist_3d = calculate_3d_distance(p_init_3d, p_max_3d) * 100.0 # cm
                            
                            print("\\n" + "="*50)
                            print(f"MOVEMENT {curr_movement_num}")
                            print("="*50)
                            print(f"Original Frame Resolution: {w}x{h}")
                            print(f"Moving ankle: {moving_ankle}")
                            print(f"Start frame: {start_frame_idx}")
                            print(f"Maximum frame: {max_frame_idx}")
                            print(f"Maximum timestamp: {max_timestamp:.2f} sec")
                            print(f"Initial raw MediaPipe normalized coords:\\n{ref_raw}")
                            print(f"Maximum raw MediaPipe normalized coords:\\n{max_raw}")
                            print(f"Initial pixel position:\\n{ref_pos}")
                            print(f"Maximum pixel position:\\n{max_pos}")
                            print(f"Pixel displacement:\\n{max_pixel_dist:.2f} px")
                            print(f"Initial metric depth (center, median):\\n{initial_center:.4f} m, {initial_depth:.4f} m")
                            print(f"Maximum metric depth (center, median):\\n{max_center:.4f} m, {max_depth:.4f} m")
                            print(f"Depth-axis difference:\\n{depth_diff:.4f} m")
                            print(f"Initial 3D position:\\n({p_init_3d[0]:.4f}, {p_init_3d[1]:.4f}, {p_init_3d[2]:.4f}) m")
                            print(f"Maximum 3D position:\\n({p_max_3d[0]:.4f}, {p_max_3d[1]:.4f}, {p_max_3d[2]:.4f}) m")
                            print(f"3D displacement:\\n{dist_3d:.2f} cm")
                            print("="*50)'''
content = content.replace(old_movement_print, new_movement_print)


# 8. Print for Video End
old_end_print = '''            print("MOVEMENT COMPLETE")
            depth_map = run_depth_inference(depth_model, max_frame_img, device)
            _, max_depth = sample_depth_neighborhood(depth_map, max_pos)
            
            initial_depth = l_depth_ref if moving_ankle == 'LEFT' else r_depth_ref
            depth_diff = abs(max_depth - initial_depth)
            
            ref_pos = l_ankle_ref if moving_ankle == 'LEFT' else r_ankle_ref
            p_init_3d = pixel_to_3d(ref_pos, initial_depth, CX, CY, FX, FY)
            p_max_3d = pixel_to_3d(max_pos, max_depth, CX, CY, FX, FY)
            
            dist_3d = calculate_3d_distance(p_init_3d, p_max_3d) * 100.0 # cm
            
            print("\\n" + "="*50)
            print(f"MOVEMENT {curr_movement_num}")
            print("="*50)
            print(f"Moving ankle: {moving_ankle}")
            print(f"Start frame: {start_frame_idx}")
            print(f"Maximum frame: {max_frame_idx}")
            print(f"Maximum timestamp: {max_timestamp:.2f} sec")
            print(f"Initial pixel position:\\n{ref_pos}")
            print(f"Maximum pixel position:\\n{max_pos}")
            print(f"Pixel displacement:\\n{max_pixel_dist:.2f} px")
            print(f"Initial metric depth:\\n{initial_depth:.4f} m")
            print(f"Maximum metric depth:\\n{max_depth:.4f} m")
            print(f"Depth-axis difference:\\n{depth_diff:.4f} m")
            print(f"Initial 3D position:\\n({p_init_3d[0]:.4f}, {p_init_3d[1]:.4f}, {p_init_3d[2]:.4f}) m")
            print(f"Maximum 3D position:\\n({p_max_3d[0]:.4f}, {p_max_3d[1]:.4f}, {p_max_3d[2]:.4f}) m")
            print(f"3D displacement:\\n{dist_3d:.2f} cm")
            print("="*50)'''

new_end_print = '''            print("MOVEMENT COMPLETE")
            depth_map = run_depth_inference(depth_model, max_frame_img, device)
            max_center, max_depth = sample_depth_neighborhood(depth_map, max_pos)
            
            initial_depth = l_depth_ref if moving_ankle == 'LEFT' else r_depth_ref
            initial_center = l_center_ref if moving_ankle == 'LEFT' else r_center_ref
            ref_raw = l_raw_ref if moving_ankle == 'LEFT' else r_raw_ref
            
            depth_diff = abs(max_depth - initial_depth)
            
            ref_pos = l_ankle_ref if moving_ankle == 'LEFT' else r_ankle_ref
            p_init_3d = pixel_to_3d(ref_pos, initial_depth, CX, CY, FX, FY)
            p_max_3d = pixel_to_3d(max_pos, max_depth, CX, CY, FX, FY)
            
            dist_3d = calculate_3d_distance(p_init_3d, p_max_3d) * 100.0 # cm
            
            print("\\n" + "="*50)
            print(f"MOVEMENT {curr_movement_num}")
            print("="*50)
            print(f"Original Frame Resolution: {w}x{h}")
            print(f"Moving ankle: {moving_ankle}")
            print(f"Start frame: {start_frame_idx}")
            print(f"Maximum frame: {max_frame_idx}")
            print(f"Maximum timestamp: {max_timestamp:.2f} sec")
            print(f"Initial raw MediaPipe normalized coords:\\n{ref_raw}")
            print(f"Maximum raw MediaPipe normalized coords:\\n{max_raw}")
            print(f"Initial pixel position:\\n{ref_pos}")
            print(f"Maximum pixel position:\\n{max_pos}")
            print(f"Pixel displacement:\\n{max_pixel_dist:.2f} px")
            print(f"Initial metric depth (center, median):\\n{initial_center:.4f} m, {initial_depth:.4f} m")
            print(f"Maximum metric depth (center, median):\\n{max_center:.4f} m, {max_depth:.4f} m")
            print(f"Depth-axis difference:\\n{depth_diff:.4f} m")
            print(f"Initial 3D position:\\n({p_init_3d[0]:.4f}, {p_init_3d[1]:.4f}, {p_init_3d[2]:.4f}) m")
            print(f"Maximum 3D position:\\n({p_max_3d[0]:.4f}, {p_max_3d[1]:.4f}, {p_max_3d[2]:.4f}) m")
            print(f"3D displacement:\\n{dist_3d:.2f} cm")
            print("="*50)'''

content = content.replace(old_end_print, new_end_print)


with open('sebt_video_depth_test.py', 'w') as f:
    f.write(content)
