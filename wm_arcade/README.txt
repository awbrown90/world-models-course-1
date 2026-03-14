Train a simple world model in minutes!

1. Run the interactive image encoder, should converge and match gt within 2-3 minutes
`python interactive_vqvae.py --env platformer`

2. Run the interactive world model, that uses the trained above encoder, should converge and match gt sim within 5-8 minutes
`python interactive_wm.py --env platformer`

Try other more complex games like pong, should also converge quickly, a couple more minutes longer than platformer
`python interactive_vqvae.py --env pong`
`python interactive_wm.py --env pong`
