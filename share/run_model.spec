[model]
method = option('em', 'lda', 'lsa')
debug = boolean(default=False)
show_viz = boolean(default=False)
datasource = string

[em_conf]
viz_count = integer(default=4)
topic_count = integer(default=4)
iterations = integer(default=250)
save_model = boolean(default=False)
