const BASE = 'http://localhost:8000';

export interface AnalysisResult {
	tokens: string[];
	token_ids: number[];
	n_layers: number;
	n_heads: number;
	attention: number[][][][];
	logit_lens: Record<string, [string, number][]>;
	top_predictions: [string, number][];
	/**
	 * Per-position logit-lens readout of the final layer against the top token,
	 * in logits. NOT an attribution: only the last entry is the logit the model
	 * actually predicted from; the rest are diagnostic readouts of positions that
	 * cannot contribute to it, and they do not decompose anything.
	 */
	final_layer_readout: number[];
	final_layer_readout_unit: string;
}

export async function analyzeText(text: string): Promise<AnalysisResult> {
	const res = await fetch(`${BASE}/api/analyze`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ text })
	});
	if (!res.ok) {
		const err = await res.json().catch(() => ({ detail: 'Request failed' }));
		throw new Error(err.detail ?? 'Request failed');
	}
	return res.json() as Promise<AnalysisResult>;
}

