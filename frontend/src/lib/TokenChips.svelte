<script lang="ts">
	import { createEventDispatcher } from 'svelte';

	export let tokens: string[] = [];
	export let selectedToken: number | null = null;

	const dispatch = createEventDispatcher<{ select: number | null }>();

	const PALETTE = [
		'#3b82f6', '#8b5cf6', '#10b981', '#f59e0b',
		'#ef4444', '#06b6d4', '#84cc16', '#f97316'
	];

	function chipColor(i: number): string {
		return PALETTE[i % PALETTE.length];
	}

	function toggle(i: number) {
		dispatch('select', selectedToken === i ? null : i);
	}
</script>

<div class="chips">
	{#each tokens as token, i}
		<button
			class="chip"
			class:selected={selectedToken === i}
			style="--color: {chipColor(i)}"
			on:click={() => toggle(i)}
			title="Token {i}: {token}"
		>
			{token}
		</button>
	{/each}
</div>

<style>
	.chips {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
		margin-top: 12px;
	}

	.chip {
		padding: 4px 10px;
		border-radius: 9999px;
		border: 2px solid var(--color);
		background: color-mix(in srgb, var(--color) 15%, transparent);
		color: var(--color);
		font-size: 0.8rem;
		font-weight: 600;
		transition: all 0.15s;
		white-space: pre;
	}

	.chip:hover {
		background: color-mix(in srgb, var(--color) 30%, transparent);
	}

	.chip.selected {
		background: var(--color);
		color: #fff;
	}
</style>
