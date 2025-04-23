use glam::Vec3;
use vulkano::{buffer::BufferContents, pipeline::graphics::vertex_input::Vertex};

#[derive(BufferContents, Vertex, Debug)]
#[repr(C)]
pub struct Position {
    #[format(R32G32B32_SFLOAT)]
    position: [f32; 3],
}

impl Position {
    pub fn to_vec3(&self) -> Vec3 {
        Vec3::new(self.position[0], self.position[1], self.position[2])
    }
}

pub const POSITIONS: [Position; 8] = [
    Position {
        position: [0.0, 0.0, 0.0],
    },
    Position {
        position: [1.0, 0.0, 0.0],
    },
    Position {
        position: [1.0, 0.0, 1.0],
    },
    Position {
        position: [0.0, 0.0, 1.0],
    },
    Position {
        position: [0.0, 1.0, 0.0],
    },
    Position {
        position: [1.0, 1.0, 0.0],
    },
    Position {
        position: [1.0, 1.0, 1.0],
    },
    Position {
        position: [0.0, 1.0, 1.0],
    },
];

#[derive(BufferContents, Vertex)]
#[repr(C)]
pub struct Normal {
    #[format(R32G32B32_SFLOAT)]
    normal: [f32; 3],
}

pub const NORMALS: [Normal; 8] = [
    Normal {
        normal: [1.0, 1.0, 1.0],
    },
    Normal {
        normal: [1.0, 1.0, 1.0],
    },
    Normal {
        normal: [1.0, 1.0, 1.0],
    },
    Normal {
        normal: [1.0, 1.0, 1.0],
    },
    Normal {
        normal: [1.0, 1.0, 1.0],
    },
    Normal {
        normal: [1.0, 1.0, 1.0],
    },
    Normal {
        normal: [1.0, 1.0, 1.0],
    },
    Normal {
        normal: [1.0, 1.0, 1.0],
    },
];

pub const INDICES: [u16; 36] = [
    0, 4, 5, //
    0, 1, 5, //
    1, 5, 6, //
    1, 2, 6, //
    2, 6, 7, //
    2, 3, 7, //
    2, 4, 7, //
    0, 2, 4, //
    0, 1, 2, //
    0, 2, 3, //
    4, 5, 6, //
    4, 6, 7,
];
