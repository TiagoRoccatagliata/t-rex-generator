const canvas = document.getElementById('game');
const ctx = canvas.getContext('2d');
const chartCanvas = document.getElementById('chart');
const chartCtx = chartCanvas.getContext('2d');

const W = 800, H = 300;
const GROUND_Y = 240;
const DINO_W = 18, DINO_H = 22;
const GRAVITY = 0.7;
const JUMP_V = -11.5;

const STATE_SIZE = 5;
const ACTION_SIZE = 2;
const GAMMA = 0.95;
const EPSILON_START = 1.0;
const EPSILON_MIN = 0.02;
const EPSILON_DECAY = 0.995;
const LEARNING_RATE = 0.001;
const MEMORY_MAX = 5000;
const BATCH_SIZE = 32;
const TRAIN_EVERY = 5;
const TARGET_UPDATE_EVERY = 50;

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

class DQNAgent {
    constructor() {
        this.gamma = GAMMA;
        this.epsilon = EPSILON_START;
        this.memory = [];
        this.model = this.buildModel();
        this.targetModel = this.buildModel();
        this.syncTarget();
        this.isTraining = false;
        this.trainSteps = 0;
        this.lastLoss = 0;
    }
    buildModel() {
        const m = tf.sequential();
        m.add(tf.layers.dense({ inputShape: [STATE_SIZE], units: 24, activation: 'relu' }));
        m.add(tf.layers.dense({ units: 24, activation: 'relu' }));
        m.add(tf.layers.dense({ units: ACTION_SIZE, activation: 'linear' }));
        m.compile({ optimizer: tf.train.adam(LEARNING_RATE), loss: 'meanSquaredError' });
        return m;
    }
    syncTarget() {
        this.targetModel.setWeights(this.model.getWeights());
    }
    act(state) {
        if (Math.random() < this.epsilon) {
            return Math.random() < 0.15 ? 1 : 0;
        }
        return tf.tidy(() => {
            const q = this.model.predict(tf.tensor2d([state]));
            return q.argMax(1).dataSync()[0];
        });
    }
    remember(s, a, r, ns, done) {
        this.memory.push({ s, a, r, ns, done });
        if (this.memory.length > MEMORY_MAX) this.memory.shift();
    }
    async train() {
        if (this.isTraining) return;
        if (this.memory.length < BATCH_SIZE) return;
        this.isTraining = true;

        const batch = [];
        for (let i = 0; i < BATCH_SIZE; i++) {
            batch.push(this.memory[Math.floor(Math.random() * this.memory.length)]);
        }
        const states = batch.map(e => e.s);
        const nextStates = batch.map(e => e.ns);

        const qCurrent = tf.tidy(() =>
            this.model.predict(tf.tensor2d(states)).arraySync()
        );
        const qNext = tf.tidy(() =>
            this.targetModel.predict(tf.tensor2d(nextStates)).arraySync()
        );

        for (let i = 0; i < batch.length; i++) {
            const { a, r, done } = batch[i];
            // Bellman: Q(s,a) = r + gamma * max Q(s',a')
            qCurrent[i][a] = done ? r : r + this.gamma * Math.max(...qNext[i]);
        }

        const xs = tf.tensor2d(states);
        const ys = tf.tensor2d(qCurrent);
        const h = await this.model.fit(xs, ys, { epochs: 1, verbose: 0 });
        this.lastLoss = h.history.loss[0];
        xs.dispose();
        ys.dispose();

        this.trainSteps++;
        if (this.trainSteps % TARGET_UPDATE_EVERY === 0) this.syncTarget();
        if (this.epsilon > EPSILON_MIN) this.epsilon *= EPSILON_DECAY;

        this.isTraining = false;
    }
}

class Dino {
    constructor(agent) {
        this.x = 60;
        this.y = GROUND_Y - DINO_H;
        this.vy = 0;
        this.alive = true;
        this.score = 0;
        this.agent = agent;
        this.color = COLOR.accent;
        this.prevState = null;
        this.prevAction = 0;
    }
    jump() { if (this.y >= GROUND_Y - DINO_H - 0.5) this.vy = JUMP_V; }
    getState(obstacles, speed) {
        let obstacle = null;
        for (const o of obstacles) { if (o.x + o.w > this.x) { obstacle = o; break; } }
        const dx = obstacle ? (obstacle.x - this.x) / W : 1;
        const ow = obstacle ? obstacle.w / 50 : 0;
        const oh = obstacle ? obstacle.h / 50 : 0;
        const v = speed / 15;
        const onGround = (this.y >= GROUND_Y - DINO_H - 0.5) ? 1 : 0;
        return [dx, ow, oh, v, onGround];
    }
    update(obstacles, speed) {
        if (!this.alive) return;

        const state = this.getState(obstacles, speed);
        const action = this.agent.act(state);
        if (action === 1) this.jump();

        this.vy += GRAVITY;
        this.y += this.vy;
        if (this.y > GROUND_Y - DINO_H) { this.y = GROUND_Y - DINO_H; this.vy = 0; }

        let collided = false;
        for (const o of obstacles) {
            if (this.x < o.x + o.w - 2 &&
                this.x + DINO_W > o.x + 2 &&
                this.y < o.y + o.h - 2 &&
                this.y + DINO_H > o.y + 2) {
                collided = true;
                break;
            }
        }

        const reward = collided ? -100 : 1;
        const nextState = this.getState(obstacles, speed);
        this.agent.remember(state, action, reward, nextState, collided);

        if (collided) { this.alive = false; return; }
        this.score++;
    }
    draw(ctx, isLeader) {
        if (!this.alive) return;
        ctx.save();
        ctx.shadowColor = COLOR.accent;
        ctx.shadowBlur = 8;
        ctx.fillStyle = COLOR.accent;
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
        ctx.shadowBlur = 0;
        ctx.fillStyle = '#1a1d23';
        ctx.fillRect(x + 14, y + 2, 2, 2);
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
        this.agent = new DQNAgent();
        this.dinos = [new Dino(this.agent)];
    }
    aliveDinos() { return this.dinos.filter(d => d.alive); }
    leader() { return this.dinos[0]; }
    allDead() { return this.dinos.every(d => !d.alive); }
    evolve() {
        const score = this.dinos[0].score;
        if (score > this.bestEver) this.bestEver = score;
        const recent = this.history.slice(-9).map(h => h.best).concat(score);
        const avg = recent.reduce((s, v) => s + v, 0) / recent.length;
        this.history.push({ gen: this.generation, best: score, avg });
        if (this.history.length > 60) this.history.shift();
        this.dinos = [new Dino(this.agent)];
        this.generation++;
    }
}

let pop = new Population();
let obstacles = [];
let baseSpeed = 6;
let currentSpeed = baseSpeed;
let frames = 0;
let globalFrames = 0;
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
    globalFrames = 0;
    currentSpeed = baseSpeed;
    nextObstacleIn = 80;
    updateHUD();
    drawChart();
}

function stepOnce() {
    frames++;
    globalFrames++;
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

    if (globalFrames % TRAIN_EVERY === 0) {
        pop.agent.train();
    }

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
    pop.leader().draw(ctx, true);
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
    el.alive.textContent = pop.agent.epsilon.toFixed(2);
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
        if (globalFrames % 8 === 0) drawChart();
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
