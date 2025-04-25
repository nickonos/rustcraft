use std::f32::consts::PI;

use glam::Vec3;

#[derive(Debug, Clone)]
pub struct World {
    pub player: Player,
    pub blocks: Vec<Block>,
}

impl World {
    pub fn new() -> Self {
        let mut blocks: Vec<Block> = Vec::new();
        blocks.push(Block::new(Vec3::new(5.0, 0.0, 0.0)));

        World {
            player: Player::new(),
            blocks,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Player {
    pub position: Vec3,
    pub rotation: f32,
}

impl Player {
    pub fn new() -> Self {
        Player {
            position: Vec3::ZERO,
            rotation: PI / 2.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Block {
    pub position: Vec3,
    pub block_type: BlockType,
}

impl Block {
    pub fn new(position: Vec3) -> Self {
        Block {
            position,
            block_type: BlockType::Dirt,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum BlockType {
    Air,
    Stone,
    Dirt,
    Grass,
}
