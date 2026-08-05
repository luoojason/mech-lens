<script lang="ts">
	import '../app.css';
	import { analyzeText, type AnalysisResult } from '$lib/api';
	import AttentionHeatmap from '$lib/AttentionHeatmap.svelte';
	import LogitLens from '$lib/LogitLens.svelte';
	import TokenChips from '$lib/TokenChips.svelte';

	let inputText = '';
	let result: AnalysisResult | null = null;
	let loading = false;
	let error: string | null = null;
	let selectedToken: number | null = null;

	async function analyze() {
		if (!inputText.trim()) return;
		loading = true;
		error = null;
		result = null;
		selectedToken = null;
		try {
			result = await analyzeText(inputText);
		} catch (e) {
			error = e instanceof Error ? e.message : 'Unknown error';
		} finally {
			loading = false;
		}
	}

	function handleKeydown(e: KeyboardEvent) {
		if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) analyze();
	}
</script>

<svelte:head>
	<title>mech-lens — transformer internals explorer</title>
</svelte:head>

<header>
	<h1>mech-lens</h1>
	<p class="tagline">browser DevTools for the inside of a transformer</p>
</header>

<main>
	<div class="layout">
		<div class="left-col">
			<section class="input-panel">
				<h3>Input</h3>
				<textarea
					bind:value={inputText}
					placeholder="The Eiffel Tower is in"
					rows="3"
					on:keydown={handleKeydown}
					disabled={loading}
				></textarea>
				<div class="input-footer">
					<button class="analyze-btn" on:click={analyze} disabled={loading || !inputText.trim()}>
						{#if loading}
							<span class="spinner"></span> Analyzing… (~2-5s on CPU)
						{:else}
							Analyze
						{/if}
					</button>
					<span class="hint">⌘+Enter to run</span>
				</div>

				{#if error}
					<div class="error">{error}</div>
				{/if}

				{#if result}
					<TokenChips
						tokens={result.tokens}
						{selectedToken}
						on:select={(e) => (selectedToken = e.detail)}
					/>
				{/if}
			</section>

			{#if result}
				<LogitLens
					logitLens={result.logit_lens}
					topPredictions={result.top_predictions}
					finalLayerReadout={result.final_layer_readout}
					tokens={result.tokens}
				/>
			{:else}
				<section class="placeholder-panel">
					<div class="placeholder-inner">
						<p>Logit lens and final-layer readout appear here after analysis</p>
					</div>
				</section>
			{/if}
		</div>

		<div class="right-col">
			<AttentionHeatmap
				attention={result?.attention ?? []}
				tokens={result?.tokens ?? []}
				{selectedToken}
			/>
		</div>
	</div>
</main>

<style>
	header {
		padding: 24px 32px 0;
		border-bottom: 1px solid #1e293b;
		margin-bottom: 24px;
	}

	h1 {
		font-size: 1.75rem;
		font-weight: 800;
		color: #60a5fa;
		letter-spacing: -0.02em;
	}

	.tagline {
		font-size: 0.875rem;
		color: #475569;
		margin: 2px 0 16px;
	}

	main {
		padding: 0 24px 40px;
	}

	.layout {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 20px;
		align-items: start;
	}

	.left-col {
		display: flex;
		flex-direction: column;
		gap: 20px;
	}

	.right-col {
		position: sticky;
		top: 20px;
	}

	.input-panel {
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

	.input-footer {
		display: flex;
		align-items: center;
		gap: 12px;
		margin-top: 10px;
	}

	.analyze-btn {
		display: flex;
		align-items: center;
		gap: 8px;
		background: #3b82f6;
		color: #fff;
		border: none;
		border-radius: 8px;
		padding: 10px 22px;
		font-size: 0.95rem;
		font-weight: 600;
		transition: background 0.15s;
	}

	.analyze-btn:hover:not(:disabled) {
		background: #2563eb;
	}

	.hint {
		font-size: 0.75rem;
		color: #475569;
	}

	.spinner {
		width: 14px;
		height: 14px;
		border: 2px solid rgba(255, 255, 255, 0.3);
		border-top-color: #fff;
		border-radius: 50%;
		animation: spin 0.7s linear infinite;
		display: inline-block;
	}

	@keyframes spin {
		to { transform: rotate(360deg); }
	}

	.error {
		margin-top: 10px;
		background: #450a0a;
		border: 1px solid #b91c1c;
		color: #fca5a5;
		border-radius: 8px;
		padding: 10px 14px;
		font-size: 0.875rem;
	}

	.placeholder-panel {
		background: #1e293b;
		border-radius: 12px;
		padding: 20px;
		min-height: 200px;
		display: flex;
		align-items: center;
		justify-content: center;
	}

	.placeholder-inner {
		text-align: center;
		color: #334155;
	}

	.placeholder-inner p {
		font-size: 0.875rem;
		margin: 0;
	}

	@media (max-width: 768px) {
		.layout {
			grid-template-columns: 1fr;
		}

		.right-col {
			position: static;
		}

		main {
			padding: 0 16px 32px;
		}
	}
</style>
