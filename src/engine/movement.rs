use glam::Vec3;
use winit::keyboard::{Key, NamedKey};

const FORWARD_VECTOR: Vec3 = Vec3{ x: 0.0, y: 0.0, z: 1.0};
const BACKWARD_VECTOR: Vec3 = Vec3{ x: 0.0, y: 0.0, z: -1.0};
const LEFT_VECTOR: Vec3 = Vec3{ x: -1.0, y: 0.0, z: 0.0};
const RIGHT_VECTOR: Vec3 = Vec3{ x: 1.0, y: 0.0, z: 0.0};
const UP_VECTOR: Vec3 = Vec3{ x: 0.0, y: 1.0, z: 0.0};
const DOWN_VECTOR: Vec3 = Vec3{ x: 0.0, y: -1.0, z: 0.0};

#[repr(usize)]
#[derive(PartialEq, Eq, Hash, Clone, Copy)]
pub enum MovementDirection {
    Forward,
    Backward,
    Left,
    Right,
    Up,
    Down
}

impl MovementDirection {
    pub fn to_vec3(self) -> Vec3 {
        match self {
            MovementDirection::Forward => FORWARD_VECTOR,
            MovementDirection::Backward => BACKWARD_VECTOR,
            MovementDirection::Left => LEFT_VECTOR,
            MovementDirection::Right => RIGHT_VECTOR,
            MovementDirection::Up => UP_VECTOR,
            MovementDirection::Down => DOWN_VECTOR,
        }
    }
}

pub fn get_movement_direction(key: Key<&str>) -> Option<MovementDirection>{
    match key {
        Key::Character("w") => Some(MovementDirection::Forward),
        Key::Character("a") => Some(MovementDirection::Left),
        Key::Character("s") => Some(MovementDirection::Backward),
        Key::Character("d") => Some(MovementDirection::Right),
        Key::Named(NamedKey::Space) => Some(MovementDirection::Up),
        Key::Named(NamedKey::Control) => Some(MovementDirection::Down),
        _ => {
            None
        }
    }
}