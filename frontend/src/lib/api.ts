const BASE = 'http://localhost:8000';

export interface AnalysisResult {
	tokens: string[];
	token_ids: number[];
	n_layers: number;
	n_heads: number;
	attention: number[][][][];
	logit_lens: Record<string, [string, number][]>;
	top_predictions: [string, number][];
	logit_attribution: number[];
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

