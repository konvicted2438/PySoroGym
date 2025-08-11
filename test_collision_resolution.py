import numpy as np
from PysoroGym.Body import Body
from PysoroGym.Shape import Box, Plane
from PysoroGym.collision.collision import detect_collision
from PysoroGym.collision_resolution import ContactManifold, ContactSolver, SLOP
from PysoroGym.math_utils import q_from_axis_angle

def setup_bodies(box_pos, box_vel=(0,0,0), ground_is_plane=True):
    """Helper to create a box and a ground body."""
    if ground_is_plane:
        ground_body = Body(body_type=Body.STATIC)
        ground_shape = Plane(size=[20, 20])
        ground_body.add_shape(ground_shape)
    else: # Use a large box for ground
        ground_body = Body(body_type=Body.STATIC)
        ground_shape = Box(half_extents=[10, 0.5, 10])
        ground_body.add_shape(ground_shape, offset=(0, -0.5, 0))

    box_body = Body(mass=1.0, position=box_pos)
    box_body.vel = np.asarray(box_vel, dtype=float)
    box_shape = Box(half_extents=[0.5, 0.5, 0.5])
    box_body.add_shape(box_shape)
    
    # Set material properties on the colliders
    ground_body.shapes[0].material.friction = 0.6
    ground_body.shapes[0].material.elasticity = 0.2
    box_body.shapes[0].material.friction = 0.4
    box_body.shapes[0].material.elasticity = 0.5

    return ground_body, box_body

def test_static_penetration_resolution():
    """Test single-shot penetration resolution using real colliders."""
    print("=== Static Penetration Resolution Test ===")
    
    # Box starts penetrating the ground plane (box center at y=0.4, half-height=0.5)
    ground, box = setup_bodies(box_pos=(0, 0.4, 0))
    
    # Manually create the contact manifold for this specific test
    manifold = ContactManifold(ground.shapes[0], box.shapes[0])
    penetration_depth = (ground.pos[1] + 0) - (box.pos[1] - 0.5) # Plane y=0, box bottom y=-0.1
    manifold.add_contact(
        normal=np.array([0, 1, 0]),
        depth=penetration_depth,
        world_point_a=np.array([0, 0, 0]),
        world_point_b=np.array([0, -0.1, 0])
    )
    
    solver = ContactSolver(use_split_impulse=True)
    dt = 1/120
    
    print(f"Initial: pos_y={box.pos[1]:.6f}, depth={penetration_depth:.6f}")
    
    # Solve with multiple iterations
    solver.solve([manifold], dt, iterations=10)
    
    # After solving, the box position should be corrected
    final_depth = (ground.pos[1] + 0) - (box.pos[1] - 0.5)
    print(f"Final: pos_y={box.pos[1]:.6f}, depth={final_depth:.6f}")
    
    success = final_depth <= SLOP
    if success:
        print("✓ SUCCESS: Penetration resolved to acceptable level")
    else:
        print(f"✗ FAIL: Penetration {final_depth:.6f} > SLOP {SLOP}")
    
    return success

def test_dynamic_impact():
    """Test dynamic body impacting ground with restitution."""
    print("\n=== Dynamic Impact Test ===")
    ground, box = setup_bodies(box_pos=(0, 2.0, 0), box_vel=(0, -5.0, 0))
    
    solver = ContactSolver(use_split_impulse=True)
    dt = 1/120
    
    print(f"Initial: pos_y={box.pos[1]:.4f}, vel_y={box.vel[1]:.4f}")
    
    max_bounces = 0
    last_vel_y = box.vel[1]
    
    for step in range(240):  # 2 seconds
        box.vel[1] -= 9.81 * dt
        box.pos += box.vel * dt
        
        contact = detect_collision(ground.shapes[0], box.shapes[0])
        if contact:
            manifold = ContactManifold(contact.collider_a, contact.collider_b)
            manifold.add_contact(contact.normal, contact.depth, contact.contact_a, contact.contact_b)
            solver.solve([manifold], dt, iterations=10)
            
            if box.vel[1] > 0 and last_vel_y <= 0:
                max_bounces += 1
                print(f"  Bounce {max_bounces}: vel_y={box.vel[1]:.4f} at pos_y={box.pos[1]:.4f}")
            last_vel_y = box.vel[1]
            
    print(f"Final: pos_y={box.pos[1]:.4f}, vel_y={box.vel[1]:.4f}")
    print(f"Total bounces: {max_bounces}")
    
    success = max_bounces >= 2 and abs(box.pos[1] - 0.5) < 0.1
    if success:
        print("✓ SUCCESS: Body bounced as expected")
    else:
        print("✗ FAIL: Body did not bounce properly")
    
    return success

def test_resting_stability():
    """Test that a resting body stays stable."""
    print("\n=== Resting Stability Test ===")
    ground, box = setup_bodies(box_pos=(0, 0.5, 0)) # Exactly at rest on the plane
    
    solver = ContactSolver(use_split_impulse=True)
    dt = 1/120
    initial_y = box.pos[1]
    
    for step in range(120):  # 1 second
        box.vel[1] -= 9.81 * dt
        box.pos += box.vel * dt
        
        contact = detect_collision(ground.shapes[0], box.shapes[0])
        if contact:
            manifold = ContactManifold(contact.collider_a, contact.collider_b)
            manifold.add_contact(contact.normal, contact.depth, contact.contact_a, contact.contact_b)
            solver.solve([manifold], dt, iterations=5)
    
    drift = abs(box.pos[1] - initial_y)
    print(f"Initial: y={initial_y:.6f}")
    print(f"Final: y={box.pos[1]:.6f}, vel_y={box.vel[1]:.6f}")
    print(f"Drift: {drift:.6f}")
    
    success = drift < 0.01 and abs(box.vel[1]) < 0.1
    if success:
        print("✓ SUCCESS: Body remains stable at rest")
    else:
        print("✗ FAIL: Body drifted or gained velocity")
    
    return success

def test_friction_on_slope():
    """Test that friction brings a sliding box to a stop on a slope."""
    print("\n=== Friction on a Slope Test ===")
    
    # Create a sloped ground plane (rotated 20 degrees)
    slope_angle = np.deg2rad(20)
    ground = Body(body_type=Body.STATIC)
    ground.q = q_from_axis_angle(np.array([0, 0, 1]), slope_angle)
    ground.add_shape(Plane(size=[20, 20]))
    
    # Place a box on the slope
    start_pos = ground.transform_point(np.array([5, 0.5, 0]))
    box = Body(mass=1.0, position=start_pos)
    box.add_shape(Box(half_extents=[0.5, 0.5, 0.5]))
    
    # Set friction high enough to stop the box
    ground.shapes[0].material.friction = 0.8 # Static friction coeff > tan(20 deg) ~= 0.36
    box.shapes[0].material.friction = 0.8
    
    solver = ContactSolver(use_split_impulse=True)
    dt = 1/120
    
    print(f"Initial pos: {box.pos[0]:.4f}, {box.pos[1]:.4f}")
    
    for step in range(360): # 3 seconds
        # Apply gravity
        box.vel += np.array([0, -9.81, 0]) * dt
        box.pos += box.vel * dt
        
        contact = detect_collision(ground.shapes[0], box.shapes[0])
        if contact:
            manifold = ContactManifold(contact.collider_a, contact.collider_b)
            manifold.add_contact(contact.normal, contact.depth, contact.contact_a, contact.contact_b)
            solver.solve([manifold], dt, iterations=10)
            
    final_vel_mag = np.linalg.norm(box.vel)
    print(f"Final pos: {box.pos[0]:.4f}, {box.pos[1]:.4f}")
    print(f"Final velocity magnitude: {final_vel_mag:.6f}")
    
    success = final_vel_mag < 0.1
    if success:
        print("✓ SUCCESS: Box came to a stop due to friction")
    else:
        print("✗ FAIL: Box did not stop, friction failed")
        
    return success

def run_all_tests():
    """Run all collision resolution tests"""
    print("=" * 60)
    print("COLLISION RESOLUTION TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_static_penetration_resolution,
        test_dynamic_impact,
        test_resting_stability,
        test_friction_on_slope,
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)
    
    return all(results)

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)