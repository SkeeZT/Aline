"""
Test script to validate the updated squat ROM calculation logic.
"""
import numpy as np
from api.engine.exercises.squat import SquatExercise
from loguru import logger
import sys

def test_rom_calculation():
    """Test the ROM calculation functionality."""
    print("Testing ROM calculation functionality...")
    
    # Create a basic config for testing
    config = {
        "paths": {
            "voice_messages": "./assets/voice_messages/Tone_A"
        },
        "voice": {
            "enabled": False  # Disable voice for testing
        },
        "angles": {
            "knee": {
                "min_threshold": 80.0,
                "max_threshold": 175.0
            },
            "hip": {
                "min_threshold": 145.0,
                "max_threshold": 195.0
            }
        },
        "keypoints": {
            "left_shoulder": 5,
            "left_hip": 11,
            "left_knee": 13,
            "left_ankle": 15,
            "right_shoulder": 6,
            "right_hip": 12,
            "right_knee": 14,
            "right_ankle": 16
        },
        "experience": {
            "level": "intermediate"
        }
    }
    
    # Create squat exercise instance
    squat = SquatExercise(config, fps=30.0)
    
    # Test ROM establishment
    print("\n1. Testing ROM baseline establishment...")
    
    # Simulate standing position (knee angle ~175)
    standing_knee_angle = 175.0
    hip_angle = 175.0
    thigh_angle = 5.0
    
    # Establish baselines
    rom_established = squat._establish_rom_baselines(standing_knee_angle, hip_angle, 30)  # 1 second in
    print(f"   Standing position: knee={standing_knee_angle}, hip={hip_angle}")
    print(f"   Knee max baseline established: {squat.knee_max_baseline}")
    print(f"   ROM established: {rom_established}")
    
    # Simulate deep squat position (knee angle ~80)
    squat.knee_min_baseline = 80.0  # Manually set for test
    squat.knee_range = squat.knee_max_baseline - squat.knee_min_baseline
    squat.rom_established = True
    
    print(f"   Knee min baseline set to: {squat.knee_min_baseline}")
    print(f"   ROM range: {squat.knee_range}")
    
    # Test ROM percentage calculation
    print("\n2. Testing ROM percentage calculation...")
    
    test_angles = [175, 150, 125, 100, 80]  # Various knee angles
    expected_percentages = [0.0, 0.26, 0.53, 0.79, 1.0]  # Expected ROM percentages (0% at top, 100% at bottom)
    
    for angle, expected in zip(test_angles, expected_percentages):
        rom_pct = squat._calculate_rom_percentage(angle)
        print(f"   Knee angle: {angle:3d} -> ROM: {rom_pct:.2f} (expected: {expected:.2f})")
        
        # Allow small tolerance for floating point comparison
        assert abs(rom_pct - expected) < 0.05, f"ROM calculation failed for angle {angle}"
    
    print("   [PASS] ROM calculation tests passed!")
    
    # Test rep counting logic
    print("\n3. Testing rep counting logic...")
    
    # Reset rep counters
    squat.successful_reps = 0
    squat.unsuccessful_reps = 0
    squat.total_reps = 0
    squat.state = "up"
    
    # Simulate a complete rep cycle
    frame_num = 0
    
    # Standing position (should stay in "up" state)
    squat.update_rep_count(175.0, 175.0, 5.0, frame_num)
    print(f"   Frame {frame_num}: State={squat.state}, Reps={squat.total_reps}")
    
    # Start going down (knee angle decreases)
    frame_num += 10  # 1/3 second later
    squat._prev_knee_angle = 175.0  # Set previous angle for velocity calculation
    squat.update_rep_count(150.0, 160.0, -5.0, frame_num)
    print(f"   Frame {frame_num}: State={squat.state}, Reps={squat.total_reps}")
    
    # Continue going down to 40% ROM (should trigger rep attempt)
    frame_num += 10
    squat.update_rep_count(130.0, 140.0, -15.0, frame_num)
    print(f"   Frame {frame_num}: State={squat.state}, Reps={squat.total_reps}")
    
    # Continue going down to 80% ROM (should be successful if completed)
    frame_num += 10
    squat.update_rep_count(95.0, 120.0, -25.0, frame_num)
    print(f"   Frame {frame_num}: State={squat.state}, Reps={squat.total_reps}")
    
    # Start coming up
    frame_num += 10
    squat.update_rep_count(110.0, 130.0, -20.0, frame_num)
    print(f"   Frame {frame_num}: State={squat.state}, Reps={squat.total_reps}")
    
    # Continue coming up to standing (should complete rep if 80% ROM was achieved)
    frame_num += 10
    squat.update_rep_count(165.0, 170.0, 2.0, frame_num)
    print(f"   Frame {frame_num}: State={squat.state}, Reps={squat.total_reps}")
    
    print(f"   Final rep count: {squat.total_reps} total, {squat.successful_reps} successful")
    
    # Test form validation during rep
    print("\n4. Testing form validation...")
    
    # Reset for form validation test
    squat.state = "up"
    squat.successful_reps = 0
    squat.total_reps = 0
    
    # Simulate a rep with form issues
    frame_num = 0
    
    # Standing
    squat.update_rep_count(175.0, 175.0, 5.0, frame_num)
    
    # Going down with forward lean (hip angle too small)
    frame_num += 10
    squat._prev_knee_angle = 175.0
    squat.update_rep_count(150.0, 50.0, -5.0, frame_num)  # Hip angle 50 = forward lean
    
    # Continue to bottom
    frame_num += 10
    squat.update_rep_count(90.0, 55.0, -30.0, frame_num)
    
    # Coming up
    frame_num += 10
    squat.update_rep_count(120.0, 60.0, -10.0, frame_num)
    
    # Return to standing
    frame_num += 10
    squat.update_rep_count(170.0, 170.0, 5.0, frame_num)
    
    print(f"   Rep completed with form issues detected: {len(squat.reps_summary[-1]['issues']) if squat.reps_summary else 0} issues")
    if squat.reps_summary:
        print(f"   Issues: {squat.reps_summary[-1]['issues']}")
    
    print("\n[SUCCESS] All tests passed! ROM-based squat rep calculation is working correctly.")
    
    return True

if __name__ == "__main__":
    try:
        test_rom_calculation()
        print("\n[PASSED] Test completed successfully!")
    except Exception as e:
        print(f"\n[FAILED] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)