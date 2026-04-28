<script lang="ts">
	import * as d3 from 'd3';

	export let attention: number[][][][] = [];
	export let tokens: string[] = [];
	export let selectedToken: number | null = null;

	let selectedLayer = 0;
	let selectedHead = 0;
	let svgContainer: HTMLDivElement;
	let tooltip: HTMLDivElement;

	const CELL = 44;
	const MARGIN = { top: 70, right: 20, bottom: 20, left: 70 };

	$: n_layers = attention.length;
	$: n_heads = attention[0]?.length ?? 0;

	function truncate(s: string, n = 7): string {
		return s.length > n ? s.slice(0, n) + '…' : s;
	}

	function draw() {
		if (!svgContainer || !attention.length || !tokens.length) return;

		const matrix: number[][] = attention[selectedLayer]?.[selectedHead] ?? [];
		if (!matrix.length) return;

		const n = matrix.length;
		const W = CELL * n + MARGIN.left + MARGIN.right;
		const H = CELL * n + MARGIN.top + MARGIN.bottom;

		d3.select(svgContainer).selectAll('*').remove();

		const svg = d3.select(svgContainer)
			.append('svg')
			.attr('width', W)
			.attr('height', H);

		const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

		const color = d3.scaleLinear<string>()
			.domain([0, 1])
			.range(['#ffffff', '#1e40af'])
			.clamp(true);

		for (let row = 0; row < n; row++) {
			for (let col = 0; col < n; col++) {
				const val = matrix[row][col];
				const isAmber = selectedToken !== null && row === selectedToken;

				g.append('rect')
					.attr('x', col * CELL)
					.attr('y', row * CELL)
					.attr('width', CELL - 2)
					.attr('height', CELL - 2)
					.attr('rx', 3)
					.attr('fill', isAmber ? '#f59e0b' : color(val))
					.on('mouseover', (event: MouseEvent) => {
						const src = tokens[row] ?? `T${row}`;
						const dst = tokens[col] ?? `T${col}`;
						d3.select(tooltip)
							.style('display', 'block')
							.text(`${src} → ${dst}: ${val.toFixed(3)}`);
						moveTooltip(event);
					})
					.on('mousemove', moveTooltip)
					.on('mouseout', () => {
						d3.select(tooltip).style('display', 'none');
					});
			}
		}

		// Row labels (attending from)
		tokens.forEach((tok, i) => {
			g.append('text')
				.attr('x', -8)
				.attr('y', i * CELL + CELL / 2)
				.attr('text-anchor', 'end')
				.attr('dominant-baseline', 'middle')
				.attr('font-size', 11)
				.attr('fill', selectedToken === i ? '#f59e0b' : '#94a3b8')
				.attr('font-weight', selectedToken === i ? '700' : '400')
				.text(truncate(tok));
		});

		// Column labels (attending to) — rotated
		tokens.forEach((tok, i) => {
			g.append('text')
				.attr('transform', `translate(${i * CELL + CELL / 2}, -8) rotate(-45)`)
				.attr('text-anchor', 'start')
				.attr('font-size', 11)
				.attr('fill', '#94a3b8')
				.text(truncate(tok));
		});

		// Axis labels
		svg.append('text')
			.attr('x', MARGIN.left - 8)
			.attr('y', MARGIN.top / 2)
			.attr('font-size', 11)
			.attr('fill', '#64748b')
			.attr('text-anchor', 'middle')
			.text('from ↓');

		svg.append('text')
			.attr('x', MARGIN.left + (n * CELL) / 2)
			.attr('y', 14)
			.attr('font-size', 11)
			.attr('fill', '#64748b')
			.attr('text-anchor', 'middle')
			.text('→ to');
	}

	function moveTooltip(event: MouseEvent) {
		d3.select(tooltip)
			.style('left', `${event.pageX + 12}px`)
			.style('top', `${event.pageY - 32}px`);
	}

	$: if (svgContainer && attention.length > 0) {
		draw();
	}

	// Redraw when these change
	$: selectedLayer, selectedHead, selectedToken, draw();

</script>

<section>
	<div class="panel-header">
		<h3>Attention Patterns</h3>
		<div class="controls">
			<label>
				Layer
				<select bind:value={selectedLayer}>
					{#each Array.from({ length: n_layers || 12 }, (_, i) => i) as i}
						<option value={i}>Layer {i}</option>
					{/each}
				</select>
			</label>
			<label>
				Head
				<select bind:value={selectedHead}>
					{#each Array.from({ length: n_heads || 12 }, (_, i) => i) as i}
						<option value={i}>Head {i}</option>
					{/each}
				</select>
			</label>
		</div>
	</div>

	<div class="heatmap-wrap">
		{#if attention.length === 0}
			<div class="placeholder">Run analysis to see attention patterns</div>
		{:else}
			<div bind:this={svgContainer} class="svg-container"></div>
		{/if}
	</div>
</section>

<div bind:this={tooltip} class="tooltip"></div>

<style>
	section {
		background: #1e293b;
		border-radius: 12px;
		padding: 20px;
	}

	.panel-header {
		display: flex;
		align-items: flex-start;
		justify-content: space-between;
		flex-wrap: wrap;
		gap: 10px;
		margin-bottom: 16px;
	}

	h3 {
		font-size: 1rem;
		font-weight: 700;
		color: #94a3b8;
		text-transform: uppercase;
		letter-spacing: 0.08em;
	}

	.controls {
		display: flex;
		gap: 12px;
	}

	label {
		display: flex;
		align-items: center;
		gap: 6px;
		font-size: 0.8rem;
		color: #94a3b8;
	}

	.heatmap-wrap {
		overflow-x: auto;
	}

	.svg-container {
		display: inline-block;
	}

	.placeholder {
		color: #334155;
		font-size: 0.9rem;
		padding: 40px 0;
		text-align: center;
	}

	.tooltip {
		display: none;
		position: fixed;
		background: #0f172a;
		color: #e2e8f0;
		border: 1px solid #334155;
		border-radius: 6px;
		padding: 6px 10px;
		font-size: 0.8rem;
		pointer-events: none;
		z-index: 1000;
		white-space: nowrap;
	}
</style>
