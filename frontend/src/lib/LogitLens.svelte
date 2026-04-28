<script lang="ts">
	export let logitLens: Record<string, [string, number][]> = {};
	export let topPredictions: [string, number][] = [];
	export let logitAttribution: number[] = [];
	export let tokens: string[] = [];

	const N_LAYERS = 12;

	function pct(p: number): string {
		return `${(p * 100).toFixed(1)}%`;
	}

	function cellOpacity(p: number): string {
		return `rgba(59, 130, 246, ${Math.min(p * 4, 0.85)})`;
	}

	$: layers = Array.from({ length: N_LAYERS }, (_, i) => ({
		idx: i,
		preds: logitLens[`layer_${i}`] ?? []
	}));

	$: maxAttribution = Math.max(...logitAttribution, 1e-9);
</script>

<section>
	<h3>Logit Lens</h3>

	{#if layers.some(l => l.preds.length > 0)}
		<div class="table-wrap">
			<table>
				<thead>
					<tr>
						<th>Layer</th>
						<th>#1</th>
						<th>#2</th>
						<th>#3</th>
					</tr>
				</thead>
				<tbody>
					{#each layers as { idx, preds }}
						<tr class:final={idx === 11}>
							<td class="layer-label">{idx === 11 ? '11 ★' : idx}</td>
							{#each [0, 1, 2] as col}
								{#if preds[col]}
									<td
										class="pred-cell"
										style="background: {cellOpacity(preds[col][1])}"
									>
										<span class="token-name">{preds[col][0]}</span>
										<span class="prob">{pct(preds[col][1])}</span>
									</td>
								{:else}
									<td class="pred-cell empty">—</td>
								{/if}
							{/each}
						</tr>
					{/each}
				</tbody>
			</table>
		</div>

		{#if topPredictions.length > 0}
			<p class="top-preds">
				Top predictions:
				{topPredictions.slice(0, 5).map(([t, p]) => `${t} ${pct(p)}`).join(', ')}
			</p>
		{/if}
	{/if}

	{#if logitAttribution.length > 0}
		<div class="attribution">
			<h4>Token contribution to top prediction</h4>
			<div class="bars">
				{#each logitAttribution as score, i}
					<div class="bar-group">
						<span class="bar-label">{tokens[i] ?? `T${i}`}</span>
						<div class="bar-track">
							<div
								class="bar-fill"
								style="width: {(score / maxAttribution) * 100}%"
							></div>
						</div>
						<span class="bar-pct">{pct(score)}</span>
					</div>
				{/each}
			</div>
		</div>
	{/if}
</section>

<style>
	section {
		background: #1e293b;
		border-radius: 12px;
		padding: 20px;
	}

	h3 {
		font-size: 1rem;
		font-weight: 700;
		color: #94a3b8;
		text-transform: uppercase;
		letter-spacing: 0.08em;
		margin-bottom: 14px;
	}

	h4 {
		font-size: 0.875rem;
		font-weight: 600;
		color: #94a3b8;
		margin: 16px 0 10px;
	}

	.table-wrap {
		overflow-x: auto;
	}

	table {
		width: 100%;
		border-collapse: collapse;
		font-size: 0.85rem;
	}

	th {
		text-align: left;
		padding: 6px 10px;
		color: #64748b;
		font-weight: 600;
		border-bottom: 1px solid #334155;
	}

	td {
		padding: 5px 10px;
		border-bottom: 1px solid #0f172a22;
	}

	.layer-label {
		color: #64748b;
		font-variant-numeric: tabular-nums;
		white-space: nowrap;
		width: 52px;
	}

	.pred-cell {
		border-radius: 4px;
		transition: background 0.2s;
	}

	.pred-cell .token-name {
		font-weight: 600;
		margin-right: 4px;
		color: #e2e8f0;
		font-family: monospace;
	}

	.pred-cell .prob {
		font-size: 0.75rem;
		color: #94a3b8;
	}

	.pred-cell.empty {
		color: #334155;
	}

	tr.final {
		background: #1e3a5f;
	}

	tr.final .layer-label {
		color: #60a5fa;
		font-weight: 700;
	}

	.top-preds {
		margin-top: 10px;
		font-size: 0.85rem;
		color: #94a3b8;
	}

	.attribution {
		margin-top: 4px;
	}

	.bars {
		display: flex;
		flex-direction: column;
		gap: 8px;
	}

	.bar-group {
		display: grid;
		grid-template-columns: 80px 1fr 48px;
		align-items: center;
		gap: 8px;
	}

	.bar-label {
		font-family: monospace;
		font-size: 0.8rem;
		color: #94a3b8;
		text-align: right;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.bar-track {
		background: #0f172a;
		border-radius: 4px;
		height: 18px;
		overflow: hidden;
	}

	.bar-fill {
		height: 100%;
		background: linear-gradient(90deg, #3b82f6, #60a5fa);
		border-radius: 4px;
		transition: width 0.3s ease;
	}

	.bar-pct {
		font-size: 0.75rem;
		color: #64748b;
		text-align: right;
	}
</style>
