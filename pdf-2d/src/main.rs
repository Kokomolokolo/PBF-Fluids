use macroquad::prelude::*;
use std::collections::HashMap;

const PARTICLE_RADIUS: f32 = 0.1;
const H: f32 = PARTICLE_RADIUS * 4.0; // Smoothing radius
const REST_DENSITY: f32 = 1.0;
const EPSILON: f32 = 0.0001;
const SOLVER_ITERATIONS: usize = 4;
const GRAVITY: Vec2 = Vec2::new(0.0, -9.81);
const DT: f32 = 0.016;


/// Poly6 Kernel: Berechnet Gewicht basierend auf Distanz
/// Verwendet für: Dichteberechnung
/// Formula: W(r,h) = (315/(64πh^9)) * (h² - r²)³  für 0 ≤ r ≤ h
fn poly6_kernel(r: f32, h: f32) -> f32 {
    if r >= 0.0 && r <= h {
        let h2 = h * h;
        let h9 = h2 * h2 * h2 * h2 * h;
        let r2 = r * r;
        315.0 / (64.0 * std::f32::consts::PI * h9) * (h2 - r2).powi(3)
    } else {
        0.0
    }
}

/// Spiky Kernel Gradient: Berechnet Richtung und Stärke des "Drucks"
/// Verwendet für: Kraft zwischen Partikeln
/// Formula: ∇W(r,h) = -(45/(πh^6)) * (h-r)² * (r/|r|)
fn spiky_gradient(r_vec: Vec2, h: f32) -> Vec2 {
    let r = r_vec.length();
    if r > 0.0 && r <= h {
        let h6 = h * h * h * h * h * h;
        let factor = -45.0 / (std::f32::consts::PI * h6) * (h - r).powi(2);
        r_vec.normalize() * factor
    } else {
        Vec2::ZERO
    }
}


struct Particle {
    pos: Vec2,
    vel: Vec2,
    predicted_pos: Vec2,
    density: f32,
    lambda: f32,
}

struct SpatialHash {
    cell_size: f32,
    grid: HashMap<(i32, i32), Vec<usize>>,
}
impl SpatialHash {
    fn new(cell_size: f32) -> Self {
        Self {
            cell_size,
            grid: HashMap::new(),
        }
    }

    fn cell_coord(&self, pos: Vec2) -> (i32, i32) {
        (
            (pos.x / self.cell_size).floor() as i32,
            (pos.y / self.cell_size).floor() as i32,
        )
    }

    fn clear(&mut self) {
        self.grid.clear();
    }

    fn insert(&mut self, idx: usize, pos: Vec2) {
        let cell = self.cell_coord(pos);
        self.grid.entry(cell).or_insert_with(Vec::new).push(idx);
    }

    fn query_neighbors(&self, pos: Vec2) -> Vec<usize> {
        let center = self.cell_coord(pos);
        let mut neighbors = Vec::new();

        for dx in -1..=1 {
            for dy in -1..=1 {
                let cell = (center.0 + dx, center.1 + dy);
                if let Some(indices) = self.grid.get(&cell) {
                    neighbors.extend(indices);
                }
            }
        }
        neighbors
    }
}

struct FluidSimulation {
    particles: Vec<Particle>,
    spatial_hash: SpatialHash,
    bounds: Vec2,
}

impl FluidSimulation {
    fn new(bounds: Vec2) -> Self {
        let mut particles = Vec::new();
        
        let spacing = PARTICLE_RADIUS * 2.0;
        let start_x = bounds.x * 0.2;
        let start_y = bounds.y * 0.3;
        let width = bounds.x * 0.3;
        let height = bounds.y * 0.4;

        let mut y = start_y;
        while y < start_y + height {
            let mut x = start_x;
            while x < start_x + width {
                particles.push(Particle {
                    pos: Vec2::new(x, y),
                    vel: Vec2::ZERO,
                    predicted_pos: Vec2::ZERO,
                    density: REST_DENSITY,
                    lambda: 0.0,
                });
                x += spacing;
            }
            y += spacing;
        }

        Self {
            particles,
            spatial_hash: SpatialHash::new(H),
            bounds,
        }
    }
    fn step(&mut self) {
        // Gravitation anwenden und die pre.. pos errechnen
        for p in &mut self.particles {
            p.vel += GRAVITY * DT;
            p.predicted_pos = p.pos + p.vel * DT;
        }
        // Spatial Hash leeren und neu bauen
        self.spatial_hash.clear();
        for (i, p) in self.particles.iter().enumerate() {
            self.spatial_hash.insert(i, p.predicted_pos);
        }

        for _ in 0..SOLVER_ITERATIONS {
            for i in 0..self.particles.len() {
                let neighbors = self.spatial_hash.query_neighbors(self.particles[i].predicted_pos);

                let mut density = 0.0;
                for &j in &neighbors {
                    let r = (self.particles[i].predicted_pos - self.particles[j].predicted_pos).length();
                    density += poly6_kernel(r, H) // H ist der smoothing radius :/
                }
                self.particles[i].density = density;

                // Constaint, die Bedinung, welche erfüllt sein muss
                let constraint = density / REST_DENSITY - 1.0;

                // Die "Empfindlichkeit" wie stark reagiert die Dichte Bewegung
                let mut sum_gradient_squared = 0.0;
                for &j in &neighbors {
                    let r_vec = self.particles[i].predicted_pos - self.particles[j].predicted_pos;
                    let gradient = spiky_gradient(r_vec, H) / REST_DENSITY;
                    sum_gradient_squared += gradient.length_squared();
                }
            
                self.particles[i].lambda = -constraint / (sum_gradient_squared + EPSILON);
            }
        
            let mut deltas = vec![Vec2::ZERO; self.particles.len()];

            for i in 0..self.particles.len() {
                let neighbors = self.spatial_hash.query_neighbors(self.particles[i].predicted_pos);
                
                for &j in &neighbors {
                    if i == j { continue }

                    let r_vec = self.particles[i].predicted_pos - self.particles[j].predicted_pos;
                    let gradient = spiky_gradient(r_vec, H);

                    let delta = (self.particles[i].lambda + self.particles[j].lambda) * gradient / REST_DENSITY;

                    deltas[i] += delta;
                }
            }
            for i in 0..self.particles.len() {
                self.particles[i].predicted_pos += deltas[i];
            }
        }
        for p in &mut self.particles {
            p.vel = (p.predicted_pos - p.pos) / DT;
            p.pos = p.predicted_pos;

            if p.pos.x < PARTICLE_RADIUS {
                p.pos.x = PARTICLE_RADIUS;
                p.vel.x *= -0.5;
            }
            if p.pos.x > self.bounds.x - PARTICLE_RADIUS {
                p.pos.x = self.bounds.x - PARTICLE_RADIUS;
                p.vel.x *= -0.5;
            }
            if p.pos.y < PARTICLE_RADIUS {
                p.pos.y = PARTICLE_RADIUS;
                p.vel.y *= -0.5;
            }
            if p.pos.y > self.bounds.y - PARTICLE_RADIUS {
                p.pos.y = self.bounds.y - PARTICLE_RADIUS;
                p.vel.y *= -0.5;
            }
        }
    }
    fn draw(&self, scale: f32) {
        for p in &self.particles {
            let screen_pos = vec2(p.pos.x * scale, screen_height() - p.pos.y * scale);
            // let color_intensity = (p.density / REST_DENSITY).min(2.0) / 2.0;
            // let color = Color::new(0.2, 0.5 + color_intensity * 0.5, 1.0, 1.0);
            draw_circle(screen_pos.x, screen_pos.y, PARTICLE_RADIUS * scale, BLUE);
        }
    }
}
#[macroquad::main("MyGame")]
async fn main() {
    let sim_bounds = Vec2::new(10.0, 8.0);
    let mut simulation = FluidSimulation::new(sim_bounds);
    loop {
        clear_background(BLACK);

        let scale = (screen_width() / sim_bounds.x).min(screen_height() / sim_bounds.y) * 0.9;

        draw_rectangle_lines(0.0, 0.0, sim_bounds.x * scale, sim_bounds.y * scale, 2.0, DARKGRAY);

        simulation.step();
        simulation.draw(scale);

        draw_text(&format!("Particles: {}", simulation.particles.len()), 10.0, 20.0, 20.0, WHITE);
        draw_text(&format!("FPS: {}", get_fps()), 10.0, 40.0, 20.0, WHITE);

        next_frame().await
    }
}