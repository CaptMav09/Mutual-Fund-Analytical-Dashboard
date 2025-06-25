from flask import Flask, render_template, request, jsonify
from mftool import Mftool
import plotly.express as px

from analysis.nav import analyze_historical_nav
from analysis.volatility import run_monte_carlo_var

app = Flask(__name__)
mf = Mftool()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/nav')
def nav_view():
    scheme_code = request.args.get('code')
    if not scheme_code:
        return jsonify({'error': 'scheme code required'}), 400
    data, success = analyze_historical_nav(scheme_code, mf)
    if not success:
        return jsonify({'error': 'unable to fetch NAV data'}), 404
    fig = px.line(data, x='Date', y='NAV', title=f'NAV Trend - {scheme_code}')
    graph_html = fig.to_html(full_html=False)
    return render_template('nav.html', graph_html=graph_html)


@app.route('/api/nav/<scheme_code>')
def nav_api(scheme_code):
    data, success = analyze_historical_nav(scheme_code, mf)
    if not success:
        return jsonify({'error': 'unable to fetch NAV data'}), 404
    return jsonify(data.to_dict(orient='records'))


@app.route('/monte_carlo/<scheme_code>')
def monte_carlo_view(scheme_code):
    num_simulations = int(request.args.get('sims', 500))
    num_days = int(request.args.get('days', 252))
    confidence = float(request.args.get('conf', 0.95))
    nav_data, success = analyze_historical_nav(scheme_code, mf)
    if not success:
        return jsonify({'error': 'unable to fetch NAV data'}), 404
    results = run_monte_carlo_var(nav_data, scheme_code, num_simulations, num_days, confidence)
    if results.get('figure') is None:
        return jsonify({'error': results.get('error')}), 500
    graph_html = results['figure'].to_html(full_html=False)
    return render_template('monte_carlo.html', graph_html=graph_html,
                           var=results['var'], es=results['expected_shortfall'])


@app.route('/api/monte_carlo/<scheme_code>')
def monte_carlo_api(scheme_code):
    num_simulations = int(request.args.get('sims', 500))
    num_days = int(request.args.get('days', 252))
    confidence = float(request.args.get('conf', 0.95))
    nav_data, success = analyze_historical_nav(scheme_code, mf)
    if not success:
        return jsonify({'error': 'unable to fetch NAV data'}), 404
    results = run_monte_carlo_var(nav_data, scheme_code, num_simulations, num_days, confidence)
    return jsonify({'var': results.get('var'),
                    'expected_shortfall': results.get('expected_shortfall')})


if __name__ == '__main__':
    app.run(debug=True)
