function quadcopter_visualization_with_controls(model_output, H, D, l)
    % Get screen dimensions
    screen_size = get(0, 'ScreenSize');
    screen_width = screen_size(3);
    screen_height = screen_size(4);
    
    % Create a larger figure to show everything
    main_fig = figure(2);
    set(main_fig, 'Position', [50, 50, screen_width-100, screen_height-150], ...
        'NumberTitle', 'off', 'Name', 'Quadcopter Visualization with Controls', ...
        'CloseRequestFcn', @figure_close_callback);
    
    % Calculate velocity magnitude
    velocity_magnitude = sqrt((model_output.xVel).^2 + (model_output.zVel).^2);
    
    % Calculate angular acceleration (θ̈) from theta_dot
    if isfield(model_output, 'theta_dot') && length(model_output.theta_dot) > 1
        dt = mean(diff(model_output.tout));
        if dt > 0
            theta_double_dot = gradient(model_output.theta_dot, dt);
        else
            theta_double_dot = zeros(size(model_output.theta_dot));
        end
    else
        % If theta_dot doesn't exist, estimate from theta
        if length(model_output.theta) > 1
            dt = mean(diff(model_output.tout));
            if dt > 0
                theta_dot_est = gradient(model_output.theta, dt);
                theta_double_dot = gradient(theta_dot_est, dt);
            else
                theta_double_dot = zeros(size(model_output.theta));
            end
        else
            theta_double_dot = zeros(size(model_output.tout));
        end
    end
    
    % Calculate rotor positions
    arm_length = 0.5;
    rotor1_X = model_output.xPos - (arm_length/2)*cos(-model_output.theta);
    rotor1_Z = model_output.zPos - (arm_length/2)*sin(-model_output.theta);
    rotor2_X = model_output.xPos + (arm_length/2)*cos(-model_output.theta);
    rotor2_Z = model_output.zPos + (arm_length/2)*sin(-model_output.theta);
    
    % Calculate interception and bounds
    interception_idx = length(model_output.tout);
    for i = 1:length(model_output.tout)
        if ((model_output.xPos(i) - model_output.target_X(i))^2 + ...
            (model_output.zPos(i) - model_output.target_Z(i))^2 <= (l)^2)
            interception_idx = i;
            break;
        end
    end
    
    % Calculate distances
    center_distance = sqrt((model_output.xPos - model_output.target_X).^2 + ...
                          (model_output.zPos - model_output.target_Z).^2);
    [min_dist, end_of_sim] = min(center_distance);
    
    % Create a 3x3 grid layout for more plots
    % Visualization takes 3x2 space (left side)
    % Time series plots take the remaining 7 subplots
    
    % ================= MAIN VISUALIZATION =================
    ax_vis = subplot(3, 5, [1 2 6 7 11 12]);  % Takes 3x2 grid space
    hold(ax_vis, 'on');
    axis(ax_vis, 'equal');
    
    % Initialize plot objects for visualization
    h_path = plot(ax_vis, nan, nan, 'k--', 'LineWidth', 1.5, 'DisplayName', 'Pursuer');
    h_body = plot(ax_vis, nan, nan, 'b-', 'LineWidth', 6);
    h_center = plot(ax_vis, nan, nan, 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'y');
    h_prop1 = plot(ax_vis, nan, nan, 'ro', 'MarkerSize', 6, 'MarkerFaceColor', 'r');
    h_prop2 = plot(ax_vis, nan, nan, 'ro', 'MarkerSize', 6, 'MarkerFaceColor', 'r');
    h_arrow = quiver(ax_vis, nan, nan, nan, nan, 0, 'Color', 'g', 'LineWidth', 2, 'MaxHeadSize', 2);
    h_target = plot(ax_vis, nan, nan, 'rx', 'LineWidth', 2, 'DisplayName', 'Evader');
    
    legend([h_target, h_path], 'Location', 'best');
    
    % Set axis limits
    margin = arm_length + 0.5;
    xmin = min(-arm_length, min(model_output.target_X)) - margin;
    xmax = max(arm_length, max(model_output.target_X)) + margin;
    zmin = min(-arm_length, min(model_output.target_Z)) - margin;
    zmax = max(arm_length, max(model_output.target_Z)) + margin;
    xlim(ax_vis, [xmin xmax]);
    ylim(ax_vis, [zmin zmax]);
    title(ax_vis, 'Quadcopter Visualization');
    xlabel(ax_vis, 'x (m)');
    ylabel(ax_vis, 'z (m)');
    grid(ax_vis, 'on');
    
    % ================= TIME SERIES PLOTS =================
    % Create 7 smaller subplots on the right side
    
    % Plot 1: Angular Velocity
    ax1 = subplot(3, 5, 3);
    h_theta_dot = plot(ax1, nan, nan, 'b-', 'LineWidth', 1.5);
    title(ax1, 'Angular Velocity (θ̇)');
    xlabel(ax1, 'Time (s)');
    ylabel(ax1, 'θ̇ (rad/s)');
    grid(ax1, 'on');
    
    % Plot 2: Angular Acceleration
    ax2 = subplot(3, 5, 4);
    h_theta_double_dot = plot(ax2, nan, nan, 'm-', 'LineWidth', 1.5);
    title(ax2, 'Angular Acceleration (θ̈)');
    xlabel(ax2, 'Time (s)');
    ylabel(ax2, 'θ̈ (rad/s²)');
    grid(ax2, 'on');
    
    % Plot 3: Velocity Magnitude
    ax3 = subplot(3, 5, 5);
    h_velocity = plot(ax3, nan, nan, 'r-', 'LineWidth', 1.5);
    title(ax3, 'Velocity Magnitude');
    xlabel(ax3, 'Time (s)');
    ylabel(ax3, 'Velocity (m/s)');
    grid(ax3, 'on');
    
    % Plot 4: Distance to Target
    ax4 = subplot(3, 5, 8);
    h_distance = plot(ax4, nan, nan, 'k-', 'LineWidth', 1.5);
    title(ax4, 'Distance to Target');
    xlabel(ax4, 'Time (s)');
    ylabel(ax4, 'Distance (m)');
    grid(ax4, 'on');
    
    % Plot 5: Orientation
    ax5 = subplot(3, 5, 9);
    h_theta = plot(ax5, nan, nan, 'c-', 'LineWidth', 1.5);
    title(ax5, 'Orientation (θ)');
    xlabel(ax5, 'Time (s)');
    ylabel(ax5, 'θ (rad)');
    grid(ax5, 'on');
    
    % Plot 6: X Position
    ax6 = subplot(3, 5, 10);
    h_xpos = plot(ax6, nan, nan, 'g-', 'LineWidth', 1.5);
    title(ax6, 'X Position');
    xlabel(ax6, 'Time (s)');
    ylabel(ax6, 'x (m)');
    grid(ax6, 'on');
    
    % Plot 7: Z Position
    ax7 = subplot(3, 5, 13);
    h_zpos = plot(ax7, nan, nan, 'm-', 'LineWidth', 1.5);
    title(ax7, 'Z Position');
    xlabel(ax7, 'Time (s)');
    ylabel(ax7, 'z (m)');
    grid(ax7, 'on');
    
    % Plot 8: X Velocity
    ax8 = subplot(3, 5, 14);
    h_xvel = plot(ax8, nan, nan, 'b-', 'LineWidth', 1.5);
    title(ax8, 'X Velocity');
    xlabel(ax8, 'Time (s)');
    ylabel(ax8, 'V_x (m/s)');
    grid(ax8, 'on');
    
    % Plot 9: Z Velocity
    ax9 = subplot(3, 5, 15);
    h_zvel = plot(ax9, nan, nan, 'r-', 'LineWidth', 1.5);
    title(ax9, 'Z Velocity');
    xlabel(ax9, 'Time (s)');
    ylabel(ax9, 'V_z (m/s)');
    grid(ax9, 'on');
    
    % Create UI controls panel at the bottom
    control_panel = uipanel('Parent', main_fig, ...
                           'Title', 'Playback Controls', ...
                           'Position', [0.01 0.01 0.98 0.08], ...
                           'FontSize', 10, ...
                           'BackgroundColor', [0.95 0.95 0.95]);
    
    % Create control buttons
    btn_width = 0.08;
    btn_height = 0.6;
    btn_spacing = 0.01;
    
    % Rewind to start button
    uicontrol('Parent', control_panel, ...
              'Style', 'pushbutton', ...
              'String', '⏮ Rewind', ...
              'Units', 'normalized', ...
              'Position', [0.01 0.2 btn_width btn_height], ...
              'Callback', @rewind_callback, ...
              'FontSize', 10);
    
    % Step backward button
    uicontrol('Parent', control_panel, ...
              'Style', 'pushbutton', ...
              'String', '◀◀ Step -10', ...
              'Units', 'normalized', ...
              'Position', [0.01+btn_width+btn_spacing 0.2 btn_width btn_height], ...
              'Callback', @step_backward_callback, ...
              'FontSize', 10);
    
    % Step backward single button
    uicontrol('Parent', control_panel, ...
              'Style', 'pushbutton', ...
              'String', '◀ Step -1', ...
              'Units', 'normalized', ...
              'Position', [0.01+2*(btn_width+btn_spacing) 0.2 btn_width btn_height], ...
              'Callback', @step_backward_single_callback, ...
              'FontSize', 10);
    
    % Play/Pause button
    play_pause_btn = uicontrol('Parent', control_panel, ...
                               'Style', 'pushbutton', ...
                               'String', '▶ Play', ...
                               'Units', 'normalized', ...
                               'Position', [0.01+3*(btn_width+btn_spacing) 0.2 btn_width btn_height], ...
                               'Callback', @play_pause_callback, ...
                               'FontSize', 10, ...
                               'BackgroundColor', [0.8 0.9 0.8]);
    
    % Step forward single button
    uicontrol('Parent', control_panel, ...
              'Style', 'pushbutton', ...
              'String', '▶ Step +1', ...
              'Units', 'normalized', ...
              'Position', [0.01+4*(btn_width+btn_spacing) 0.2 btn_width btn_height], ...
              'Callback', @step_forward_callback, ...
              'FontSize', 10);
    
    % Step forward button
    uicontrol('Parent', control_panel, ...
              'Style', 'pushbutton', ...
              'String', '▶▶ Step +10', ...
              'Units', 'normalized', ...
              'Position', [0.01+5*(btn_width+btn_spacing) 0.2 btn_width btn_height], ...
              'Callback', @step_forward_multiple_callback, ...
              'FontSize', 10);
    
    % Fast forward to end button
    uicontrol('Parent', control_panel, ...
              'Style', 'pushbutton', ...
              'String', '⏭ Fast Forward', ...
              'Units', 'normalized', ...
              'Position', [0.01+6*(btn_width+btn_spacing) 0.2 btn_width btn_height], ...
              'Callback', @fast_forward_callback, ...
              'FontSize', 10);
    
    % Speed control slider
    uicontrol('Parent', control_panel, ...
              'Style', 'text', ...
              'String', 'Speed:', ...
              'Units', 'normalized', ...
              'Position', [0.01+7*(btn_width+btn_spacing) 0.2 0.05 btn_height], ...
              'FontSize', 10, ...
              'BackgroundColor', [0.95 0.95 0.95]);
    
    speed_slider = uicontrol('Parent', control_panel, ...
                             'Style', 'slider', ...
                             'Min', 0.1, ...
                             'Max', 5, ...
                             'Value', 1, ...
                             'Units', 'normalized', ...
                             'Position', [0.01+7*(btn_width+btn_spacing)+0.05 0.2 0.1 btn_height], ...
                             'Callback', @speed_callback);
    
    % Speed display
    speed_text = uicontrol('Parent', control_panel, ...
                          'Style', 'text', ...
                          'String', '1.0x', ...
                          'Units', 'normalized', ...
                          'Position', [0.01+7*(btn_width+btn_spacing)+0.16 0.2 0.05 btn_height], ...
                          'FontSize', 10, ...
                          'BackgroundColor', [0.95 0.95 0.95]);
    
    % Current frame display
    frame_text = uicontrol('Parent', control_panel, ...
                          'Style', 'text', ...
                          'String', sprintf('Frame: 1/%d', length(model_output.tout)), ...
                          'Units', 'normalized', ...
                          'Position', [0.01+8*(btn_width+btn_spacing) 0.2 0.1 btn_height], ...
                          'FontSize', 10, ...
                          'BackgroundColor', [0.95 0.95 0.95]);
    
    % Time display
    time_text = uicontrol('Parent', control_panel, ...
                         'Style', 'text', ...
                         'String', sprintf('Time: %.2f s', model_output.tout(1)), ...
                         'Units', 'normalized', ...
                         'Position', [0.01+9*(btn_width+btn_spacing) 0.2 0.12 btn_height], ...
                         'FontSize', 10, ...
                         'BackgroundColor', [0.95 0.95 0.95]);
    
    % Distance display
    dist_text = uicontrol('Parent', control_panel, ...
                         'Style', 'text', ...
                         'String', sprintf('Distance: %.2f m', center_distance(1)), ...
                         'Units', 'normalized', ...
                         'Position', [0.01+10*(btn_width+btn_spacing) 0.2 0.12 btn_height], ...
                         'FontSize', 10, ...
                         'BackgroundColor', [0.95 0.95 0.95]);
    
    % Store data in figure's UserData
    user_data.model_output = model_output;
    user_data.H = H;
    user_data.D = D;
    user_data.l = l;
    user_data.arm_length = arm_length;
    user_data.arrow_len = 1;
    user_data.current_frame = 1;
    user_data.total_frames = length(model_output.tout);
    user_data.is_playing = false;
    user_data.animation_speed = 1;
    user_data.step_size = 1;
    
    % Store angular acceleration data
    user_data.theta_double_dot = theta_double_dot;
    
    % Visualization handles
    user_data.ax_vis = ax_vis;
    user_data.h_path = h_path;
    user_data.h_body = h_body;
    user_data.h_center = h_center;
    user_data.h_prop1 = h_prop1;
    user_data.h_prop2 = h_prop2;
    user_data.h_arrow = h_arrow;
    user_data.h_target = h_target;
    
    % Time series plot handles
    user_data.h_theta_dot = h_theta_dot;
    user_data.h_theta_double_dot = h_theta_double_dot;
    user_data.h_velocity = h_velocity;
    user_data.h_distance = h_distance;
    user_data.h_theta = h_theta;
    user_data.h_xpos = h_xpos;
    user_data.h_zpos = h_zpos;
    user_data.h_xvel = h_xvel;
    user_data.h_zvel = h_zvel;
    
    % Data
    user_data.center_distance = center_distance;
    user_data.velocity_magnitude = velocity_magnitude;
    user_data.end_of_sim = end_of_sim;
    user_data.interception_idx = interception_idx;
    
    % Control handles
    user_data.frame_text = frame_text;
    user_data.time_text = time_text;
    user_data.dist_text = dist_text;
    user_data.play_pause_btn = play_pause_btn;
    user_data.speed_slider = speed_slider;
    user_data.speed_text = speed_text;
    user_data.stop_animation = false;
    
    set(main_fig, 'UserData', user_data);
    
    % Draw first frame
    update_visualization();
    
    % Add keyboard shortcuts
    set(main_fig, 'KeyPressFcn', @keyboard_callback);
    
    % Display simulation summary
    display_simulation_summary();
    
    % Start animation in a separate function
    start_animation_loop();
    
    % Callback functions
    function rewind_callback(~, ~)
        user_data = get(main_fig, 'UserData');
        user_data.current_frame = 1;
        user_data.is_playing = false;
        set(user_data.play_pause_btn, 'String', '▶ Play');
        set(main_fig, 'UserData', user_data);
        update_visualization();
    end
    
    function step_backward_callback(~, ~)
        user_data = get(main_fig, 'UserData');
        user_data.current_frame = max(1, user_data.current_frame - 10);
        user_data.is_playing = false;
        set(user_data.play_pause_btn, 'String', '▶ Play');
        set(main_fig, 'UserData', user_data);
        update_visualization();
    end
    
    function step_backward_single_callback(~, ~)
        user_data = get(main_fig, 'UserData');
        user_data.current_frame = max(1, user_data.current_frame - 1);
        user_data.is_playing = false;
        set(user_data.play_pause_btn, 'String', '▶ Play');
        set(main_fig, 'UserData', user_data);
        update_visualization();
    end
    
    function play_pause_callback(~, ~)
        user_data = get(main_fig, 'UserData');
        user_data.is_playing = ~user_data.is_playing;
        if user_data.is_playing
            set(user_data.play_pause_btn, 'String', '⏸ Pause');
        else
            set(user_data.play_pause_btn, 'String', '▶ Play');
        end
        set(main_fig, 'UserData', user_data);
    end
    
    function step_forward_callback(~, ~)
        user_data = get(main_fig, 'UserData');
        user_data.current_frame = min(user_data.total_frames, user_data.current_frame + 1);
        user_data.is_playing = false;
        set(user_data.play_pause_btn, 'String', '▶ Play');
        set(main_fig, 'UserData', user_data);
        update_visualization();
    end
    
    function step_forward_multiple_callback(~, ~)
        user_data = get(main_fig, 'UserData');
        user_data.current_frame = min(user_data.total_frames, user_data.current_frame + 10);
        user_data.is_playing = false;
        set(user_data.play_pause_btn, 'String', '▶ Play');
        set(main_fig, 'UserData', user_data);
        update_visualization();
    end
    
    function fast_forward_callback(~, ~)
        user_data = get(main_fig, 'UserData');
        user_data.current_frame = user_data.total_frames;
        user_data.is_playing = false;
        set(user_data.play_pause_btn, 'String', '▶ Play');
        set(main_fig, 'UserData', user_data);
        update_visualization();
    end
    
    function speed_callback(source, ~)
        user_data = get(main_fig, 'UserData');
        user_data.animation_speed = source.Value;
        set(user_data.speed_text, 'String', sprintf('%.1fx', user_data.animation_speed));
        set(main_fig, 'UserData', user_data);
    end
    
    function keyboard_callback(~, event)
        user_data = get(main_fig, 'UserData');
        switch event.Key
            case 'space'
                play_pause_callback([], []);
            case 'leftarrow'
                step_backward_single_callback([], []);
            case 'rightarrow'
                step_forward_callback([], []);
            case 'home'
                rewind_callback([], []);
            case 'end'
                fast_forward_callback([], []);
            case 'comma'  % <
                step_backward_callback([], []);
            case 'period' % >
                step_forward_multiple_callback([], []);
        end
    end
    
    function update_visualization()
        user_data = get(main_fig, 'UserData');
        k = user_data.current_frame;
        
        % Extract data
        model_output = user_data.model_output;
        arm_length = user_data.arm_length;
        arrow_len = user_data.arrow_len;
        
        % Current position and orientation
        xc = model_output.xPos(k);
        zc = model_output.zPos(k);
        theta = model_output.theta(k);
        
        % Compute arm endpoints
        dx = arm_length/2 * cos(-theta);
        dz = arm_length/2 * sin(-theta);
        x1 = xc - dx;
        z1 = zc - dz;
        x2 = xc + dx;
        z2 = zc + dz;
        
        % Compute arrow direction
        tx = cos(-theta + pi/2);
        tz = sin(-theta + pi/2);
        
        % ===== UPDATE VISUALIZATION =====
        % Update plot objects
        set(user_data.h_path, 'XData', model_output.xPos(1:k), ...
                             'YData', model_output.zPos(1:k));
        set(user_data.h_body, 'XData', [x1 x2], 'YData', [z1 z2]);
        set(user_data.h_center, 'XData', xc, 'YData', zc);
        set(user_data.h_prop1, 'XData', x1, 'YData', z1);
        set(user_data.h_prop2, 'XData', x2, 'YData', z2);
        set(user_data.h_arrow, 'XData', xc, 'YData', zc, ...
                              'UData', arrow_len * tx, 'VData', arrow_len * tz);
        set(user_data.h_target, 'XData', model_output.target_X(1:k), ...
                               'YData', model_output.target_Z(1:k));
        
        % Update title
        if k == user_data.interception_idx
            title(user_data.ax_vis, sprintf('🎯 INTERCEPTION! t = %.2f s', model_output.tout(k)));
        else
            title(user_data.ax_vis, sprintf('Quadcopter Position (XZ) at t = %.2f s', model_output.tout(k)));
        end
        
        % ===== UPDATE TIME SERIES PLOTS =====
        set(user_data.h_theta_dot, 'XData', model_output.tout(1:k), ...
                                  'YData', model_output.theta_dot(1:k));
        set(user_data.h_theta_double_dot, 'XData', model_output.tout(1:k), ...
                                         'YData', user_data.theta_double_dot(1:k));
        set(user_data.h_velocity, 'XData', model_output.tout(1:k), ...
                                 'YData', user_data.velocity_magnitude(1:k));
        set(user_data.h_distance, 'XData', model_output.tout(1:k), ...
                                 'YData', user_data.center_distance(1:k));
        set(user_data.h_theta, 'XData', model_output.tout(1:k), ...
                              'YData', model_output.theta(1:k));
        set(user_data.h_xpos, 'XData', model_output.tout(1:k), ...
                             'YData', model_output.xPos(1:k));
        set(user_data.h_zpos, 'XData', model_output.tout(1:k), ...
                             'YData', model_output.zPos(1:k));
        set(user_data.h_xvel, 'XData', model_output.tout(1:k), ...
                             'YData', model_output.xVel(1:k));
        set(user_data.h_zvel, 'XData', model_output.tout(1:k), ...
                             'YData', model_output.zVel(1:k));
        
        % Update control displays
        set(user_data.frame_text, 'String', ...
            sprintf('Frame: %d/%d', k, user_data.total_frames));
        set(user_data.time_text, 'String', ...
            sprintf('Time: %.2f s', model_output.tout(k)));
        set(user_data.dist_text, 'String', ...
            sprintf('Distance: %.2f m', user_data.center_distance(k)));
        
        % Force immediate update
        drawnow limitrate;
    end
    
    function display_simulation_summary()
        fprintf('\n=== Simulation Summary ===\n');
        fprintf('H = %.2f, D = %.2f\n', H, D);
        if interception_idx < length(model_output.tout)
            fprintf('🎯 Interception at frame: %d (t = %.2f s)\n', interception_idx, model_output.tout(interception_idx));
        else
            fprintf('❌ No interception\n');
        end
        fprintf('Minimum distance: %.2f m at t = %.2f s\n', min_dist, model_output.tout(end_of_sim));
        fprintf('Final position: (%.2f, %.2f)\n', model_output.xPos(end), model_output.zPos(end));
        fprintf('Target position: (%.2f, %.2f)\n', model_output.target_X(end), model_output.target_Z(end));
        fprintf('Miss distances: center=%.2f m, rotor1=%.2f m, rotor2=%.2f m\n', ...
                center_distance(end_of_sim), ...
                sqrt((rotor1_X(end_of_sim) - model_output.target_X(end_of_sim))^2 + ...
                     (rotor1_Z(end_of_sim) - model_output.target_Z(end_of_sim))^2), ...
                sqrt((rotor2_X(end_of_sim) - model_output.target_X(end_of_sim))^2 + ...
                     (rotor2_Z(end_of_sim) - model_output.target_Z(end_of_sim))^2));
        fprintf('Max angular velocity: %.2f rad/s\n', max(abs(model_output.theta_dot)));
        fprintf('Max angular acceleration: %.2f rad/s²\n', max(abs(theta_double_dot)));
        fprintf('Max velocity magnitude: %.2f m/s\n', max(velocity_magnitude));
    end
    
    function start_animation_loop()
        % Main animation loop
        while ishandle(main_fig)
            % Get current state
            user_data = get(main_fig, 'UserData');
            
            if user_data.stop_animation
                break;
            end
            
            if user_data.is_playing && user_data.current_frame < user_data.total_frames
                % Calculate frame increment based on speed
                frame_increment = round(user_data.animation_speed);
                user_data.current_frame = min(user_data.total_frames, ...
                                             user_data.current_frame + frame_increment);
                
                % Update visualization
                update_visualization();
                
                % Save updated state
                set(main_fig, 'UserData', user_data);
                
                % Control animation speed
                pause(0.05 / user_data.animation_speed);
            else
                % Not playing, just check for button presses
                pause(0.05);
            end
        end
    end
    
    function figure_close_callback(~, ~)
        user_data = get(main_fig, 'UserData');
        user_data.stop_animation = true;
        set(main_fig, 'UserData', user_data);
        delete(main_fig);
    end
end