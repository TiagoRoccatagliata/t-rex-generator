
const rand = (a = 1, b = null) => b === null ? (Math.random() * 2 - 1) * a : a + Math.random() * (b - a);
const gaussian = () => {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
};

class NeuralNet {
    constructor(nIn = 5, nHid = 8, nOut = 1, weights = null) {
        this.nIn = nIn; this.nHid = nHid; this.nOut = nOut;
        if (weights) {
            this.w1 = weights.w1.map(r => [...r]);
            this.b1 = [...weights.b1];
            this.w2 = weights.w2.map(r => [...r]);
            this.b2 = [...weights.b2];
        } else {
            this.w1 = Array.from({ length: nHid }, () => Array.from({ length: nIn }, () => rand(1)));
            this.b1 = Array.from({ length: nHid }, () => rand(1));
            this.w2 = Array.from({ length: nOut }, () => Array.from({ length: nHid }, () => rand(1)));
            this.b2 = Array.from({ length: nOut }, () => rand(1));
        }
    }
    predict(input) {
        const h = new Array(this.nHid);
        for (let i = 0; i < this.nHid; i++) {
            let s = this.b1[i];
            for (let j = 0; j < this.nIn; j++) s += this.w1[i][j] * input[j];
            h[i] = Math.tanh(s);
        }
        const o = new Array(this.nOut);
        for (let i = 0; i < this.nOut; i++) {
            let s = this.b2[i];
            for (let j = 0; j < this.nHid; j++) s += this.w2[i][j] * h[j];
            o[i] = Math.tanh(s);
        }
        return o;
    }
    clone() { return new NeuralNet(this.nIn, this.nHid, this.nOut, { w1: this.w1, b1: this.b1, w2: this.w2, b2: this.b2 }); }
    mutate(rate = 0.08, strength = 0.4) {
        const m = v => Math.random() < rate ? v + gaussian() * strength : v;
        this.w1 = this.w1.map(r => r.map(m));
        this.b1 = this.b1.map(m);
        this.w2 = this.w2.map(r => r.map(m));
        this.b2 = this.b2.map(m);
    }
    static crossover(a, b) {
        const mix = (x, y) => Math.random() < 0.5 ? x : y;
        const child = a.clone();
        for (let i = 0; i < a.nHid; i++) {
            for (let j = 0; j < a.nIn; j++) child.w1[i][j] = mix(a.w1[i][j], b.w1[i][j]);
            child.b1[i] = mix(a.b1[i], b.b1[i]);
        }
        for (let i = 0; i < a.nOut; i++) {
            for (let j = 0; j < a.nHid; j++) child.w2[i][j] = mix(a.w2[i][j], b.w2[i][j]);
            child.b2[i] = mix(a.b2[i], b.b2[i]);
        }
        return child;
    }
}

const canvas = document.getElementById('game');
const ctx = canvas.getContext('2d');
const chartCanvas = document.getElementById('chart');
const chartCtx = chartCanvas.getContext('2d');

const W = 800, H = 300;
const GROUND_Y = 240;
const DINO_W = 18, DINO_H = 22;
const GRAVITY = 0.7;
const JUMP_V = -11.5;
const POP_SIZE = 50;
const MUTATION_RATE = 0.10;
const MUTATION_STRENGTH = 0.4;

const COLOR = {
    fg: '#ffffff',
    fg_soft: 'rgba(255,255,255,0.35)',
    accent: '#f5d547',
    cactus: '#7fb17a',
    hill: 'rgba(255,255,255,0.06)',
    cloud: 'rgba(255,255,255,0.12)'
};

function resize() { canvas.width = W; canvas.height = H; }
resize();
window.addEventListener('resize', resize);

class Dino {
    constructor(brain) {
        this.x = 60;
        this.y = GROUND_Y - DINO_H;
        this.vy = 0;
        this.alive = true;
        this.score = 0;
        this.brain = brain || new NeuralNet();
        const hue = Math.floor(Math.random() * 360);
        this.color = `hsl(${hue},20%,65%)`;
    }
    jump() { if (this.y >= GROUND_Y - DINO_H - 0.5) this.vy = JUMP_V; }
    think(obstacle, speed) {
        const dx = obstacle ? (obstacle.x - this.x) / W : 1;
        const ow = obstacle ? obstacle.w / 50 : 0;
        const oh = obstacle ? obstacle.h / 50 : 0;
        const v = speed / 15;
        const onGround = (this.y >= GROUND_Y - DINO_H - 0.5) ? 1 : 0;
        const [out] = this.brain.predict([dx, ow, oh, v, onGround]);
        if (out > 0) this.jump();
    }
    update(obstacles, speed) {
        if (!this.alive) return;
        this.vy += GRAVITY;
        this.y += this.vy;
        if (this.y > GROUND_Y - DINO_H) { this.y = GROUND_Y - DINO_H; this.vy = 0; }
        let next = null;
        for (const o of obstacles) { if (o.x + o.w > this.x) { next = o; break; } }
        this.think(next, speed);
        for (const o of obstacles) {
            if (this.x < o.x + o.w - 2 &&
                this.x + DINO_W > o.x + 2 &&
                this.y < o.y + o.h - 2 &&
                this.y + DINO_H > o.y + 2) {
                this.alive = false;
                return;
            }
        }
        this.score++;
    }
    draw(ctx, isLeader) {
        if (!this.alive) return;
        ctx.save();
        if (isLeader) {
            ctx.shadowColor = COLOR.accent;
            ctx.shadowBlur = 8;
            ctx.fillStyle = COLOR.accent;
        } else {
            ctx.globalAlpha = 0.35;
            ctx.fillStyle = this.color;
        }
        const x = Math.round(this.x), y = Math.round(this.y);
        ctx.fillRect(x + 8, y + 0, 10, 8);
        ctx.fillRect(x + 16, y + 3, 2, 2);
        ctx.fillRect(x + 2, y + 8, 14, 10);
        ctx.fillRect(x + 0, y + 8, 2, 4);
        const phase = Math.floor(this.score / 3) % 2;
        ctx.fillRect(x + 4, y + 18, 3, 4);
        ctx.fillRect(x + 11, y + 18, 3, 4);
        if (phase === 0) ctx.fillRect(x + 4, y + 22, 3, 1);
        else ctx.fillRect(x + 11, y + 22, 3, 1);
        if (isLeader) {
            ctx.shadowBlur = 0;
            ctx.fillStyle = '#1a1d23';
            ctx.fillRect(x + 14, y + 2, 2, 2);
        }
        ctx.restore();
    }
}

class Obstacle {
    constructor(x, speed) {
        this.x = x;
        const r = Math.random();
        if (r < 0.55) { this.w = 12; this.h = 26; }
        else if (r < 0.85) { this.w = 16; this.h = 34; }
        else { this.w = 28; this.h = 30; }
        this.y = GROUND_Y - this.h;
    }
    update(speed) { this.x -= speed; }
    offscreen() { return this.x + this.w < 0; }
    draw(ctx) {
        ctx.fillStyle = COLOR.cactus;
        const x = Math.round(this.x), y = Math.round(this.y), w = this.w, h = this.h;
        ctx.fillRect(x + Math.floor(w / 2) - 2, y, 4, h);
        if (w >= 12) {
            ctx.fillRect(x, y + Math.floor(h * 0.3), 4, Math.floor(h * 0.4));
            ctx.fillRect(x + w - 4, y + Math.floor(h * 0.2), 4, Math.floor(h * 0.4));
        }
        if (w >= 24) ctx.fillRect(x + Math.floor(w / 2) + 4, y + 8, 3, h - 12);
    }
}

class Population {
    constructor() {
        this.generation = 1;
        this.bestEver = 0;
        this.history = [];
        this.dinos = Array.from({ length: POP_SIZE }, () => new Dino());
    }
    aliveDinos() { return this.dinos.filter(d => d.alive); }
    leader() {
        let best = this.dinos[0];
        for (const d of this.dinos) {
            if (d.alive && d.score > best.score) best = d;
            else if (!best.alive && d.alive) best = d;
        }
        return best;
    }
    allDead() { return this.dinos.every(d => !d.alive); }
    evolve() {
        const sorted = [...this.dinos].sort((a, b) => b.score - a.score);
        const genBest = sorted[0].score;
        const avg = this.dinos.reduce((s, d) => s + d.score, 0) / this.dinos.length;
        this.history.push({ gen: this.generation, best: genBest, avg });
        if (this.history.length > 60) this.history.shift();
        if (genBest > this.bestEver) this.bestEver = genBest;
        const elites = sorted.slice(0, 2).map(d => new Dino(d.brain.clone()));
        const pool = sorted.slice(0, Math.max(6, Math.floor(POP_SIZE / 2)));
        const weightedPick = () => {
            const total = pool.reduce((s, d) => s + Math.max(1, d.score), 0);
            let r = Math.random() * total;
            for (const d of pool) { r -= Math.max(1, d.score); if (r <= 0) return d; }
            return pool[0];
        };
        const children = [];
        while (children.length < POP_SIZE - elites.length) {
            const a = weightedPick();
            const b = weightedPick();
            const brain = NeuralNet.crossover(a.brain, b.brain);
            brain.mutate(MUTATION_RATE, MUTATION_STRENGTH);
            children.push(new Dino(brain));
        }
        this.dinos = [...elites, ...children];
        this.generation++;
    }
}

let pop = new Population();
let obstacles = [];
let baseSpeed = 6;
let currentSpeed = baseSpeed;
let frames = 0;
let nextObstacleIn = 80;
let running = false;
let simSpeed = 1;
let groundOffset = 0;

function spawnObstacle() {
    const last = obstacles[obstacles.length - 1];
    const minGap = 180 + currentSpeed * 6;
    const x = Math.max(W + 20, last ? last.x + last.w + minGap + Math.random() * 200 : W + 50);
    obstacles.push(new Obstacle(x, currentSpeed));
}

function resetGame() {
    pop = new Population();
    obstacles = [];
    frames = 0;
    currentSpeed = baseSpeed;
    nextObstacleIn = 80;
    updateHUD();
    drawChart();
}

function stepOnce() {
    frames++;
    currentSpeed = baseSpeed + Math.min(8, frames / 1200);
    nextObstacleIn--;
    if (nextObstacleIn <= 0) {
        spawnObstacle();
        nextObstacleIn = 70 + Math.floor(Math.random() * 60);
    }
    for (const o of obstacles) o.update(currentSpeed);
    obstacles = obstacles.filter(o => !o.offscreen());
    for (const d of pop.dinos) d.update(obstacles, currentSpeed);
    groundOffset = (groundOffset + currentSpeed) % 20;
    if (pop.allDead()) {
        pop.evolve();
        obstacles = [];
        frames = 0;
        currentSpeed = baseSpeed;
        nextObstacleIn = 80;
    }
}

function drawGround() {
    ctx.fillStyle = COLOR.fg;
    ctx.fillRect(0, GROUND_Y + 1, W, 1);
    ctx.fillStyle = COLOR.fg_soft;
    for (let x = -groundOffset; x < W; x += 20) ctx.fillRect(x, GROUND_Y + 5, 8, 1);
    ctx.fillStyle = COLOR.hill;
    const hillOffset = (groundOffset * 0.3) % 60;
    for (let x = -hillOffset; x < W + 60; x += 60) {
        ctx.beginPath();
        ctx.moveTo(x, GROUND_Y);
        ctx.lineTo(x + 30, GROUND_Y - 18);
        ctx.lineTo(x + 60, GROUND_Y);
        ctx.fill();
    }
}

function drawCloud(x, y) {
    ctx.fillStyle = COLOR.cloud;
    ctx.fillRect(x, y, 18, 4);
    ctx.fillRect(x + 4, y - 4, 10, 4);
}
let clouds = Array.from({ length: 4 }, (_, i) => ({ x: i * 220, y: 40 + Math.random() * 60 }));

function render() {
    ctx.clearRect(0, 0, W, H);
    for (const c of clouds) {
        c.x -= currentSpeed * 0.15;
        if (c.x < -30) { c.x = W + Math.random() * 60; c.y = 30 + Math.random() * 80; }
        drawCloud(c.x, c.y);
    }
    drawGround();
    for (const o of obstacles) o.draw(ctx);
    const leader = pop.leader();
    for (const d of pop.dinos) { if (d === leader) continue; d.draw(ctx, false); }
    leader.draw(ctx, true);
}

const el = {
    gen: document.getElementById('gen'),
    best: document.getElementById('best'),
    alive: document.getElementById('alive'),
    current: document.getElementById('current'),
    pauseBtn: document.getElementById('pauseBtn'),
    resetBtn: document.getElementById('resetBtn'),
    overlay: document.getElementById('overlay'),
    startBtn: document.getElementById('startBtn'),
};

function pad(n, w = 2) { return String(n).padStart(w, '0'); }

function updateHUD() {
    el.gen.innerHTML = pad(pop.generation, 2) + '<span class="unit">/ &infin;</span>';
    el.best.innerHTML = pad(pop.bestEver, 4) + '<span class="unit">pts</span>';
    el.alive.textContent = pop.aliveDinos().length;
    el.current.textContent = pop.leader().score;
}

function drawChart() {
    const c = chartCtx, cw = chartCanvas.width = chartCanvas.offsetWidth, ch = chartCanvas.height = 74;
    c.clearRect(0, 0, cw, ch);
    c.strokeStyle = 'rgba(255,255,255,0.06)';
    c.lineWidth = 1;
    for (let i = 0; i < 4; i++) {
        const y = i * (ch / 3);
        c.beginPath(); c.moveTo(0, y); c.lineTo(cw, y); c.stroke();
    }
    if (pop.history.length < 2) {
        c.fillStyle = 'rgba(255,255,255,0.3)';
        c.font = '600 11px Nunito, sans-serif';
        c.fillText('Esperando datos…', 8, ch / 2);
        return;
    }
    const maxV = Math.max(...pop.history.map(h => h.best), 10);
    const n = pop.history.length;
    c.strokeStyle = 'rgba(255,255,255,0.35)';
    c.lineWidth = 1.3;
    c.beginPath();
    pop.history.forEach((h, i) => {
        const x = (i / (n - 1)) * cw;
        const y = ch - (h.avg / maxV) * (ch - 8) - 4;
        i === 0 ? c.moveTo(x, y) : c.lineTo(x, y);
    });
    c.stroke();
    c.strokeStyle = COLOR.accent;
    c.lineWidth = 2.5;
    c.beginPath();
    pop.history.forEach((h, i) => {
        const x = (i / (n - 1)) * cw;
        const y = ch - (h.best / maxV) * (ch - 8) - 4;
        i === 0 ? c.moveTo(x, y) : c.lineTo(x, y);
    });
    c.stroke();
    const last = pop.history[n - 1];
    const lx = cw, ly = ch - (last.best / maxV) * (ch - 8) - 4;
    c.fillStyle = COLOR.accent;
    c.beginPath(); c.arc(lx - 2, ly, 3.5, 0, Math.PI * 2); c.fill();
}

function loop() {
    if (running) {
        for (let i = 0; i < simSpeed; i++) stepOnce();
        render();
        updateHUD();
        if (frames % 8 === 0) drawChart();
    } else render();
    requestAnimationFrame(loop);
}

el.startBtn.addEventListener('click', () => {
    el.overlay.classList.add('hidden');
    running = true;
    el.pauseBtn.disabled = false;
    el.pauseBtn.textContent = 'Pausar';
});
el.pauseBtn.addEventListener('click', () => {
    running = !running;
    el.pauseBtn.textContent = running ? 'Pausar' : 'Reanudar';
});
el.resetBtn.addEventListener('click', () => {
    running = false;
    resetGame();
    el.pauseBtn.disabled = true;
    el.pauseBtn.textContent = 'Pausar';
    el.overlay.classList.remove('hidden');
});
document.querySelectorAll('.btn.speed').forEach(b => {
    b.addEventListener('click', () => {
        document.querySelectorAll('.btn.speed').forEach(x => x.classList.remove('active'));
        b.classList.add('active');
        simSpeed = parseInt(b.dataset.speed, 10);
    });
});
window.addEventListener('keydown', e => {
    if (e.code === 'Space' && !el.pauseBtn.disabled) {
        e.preventDefault();
        el.pauseBtn.click();
    }
});

updateHUD();
drawChart();
loop();