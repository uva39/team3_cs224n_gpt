### loss를 mean방식 변경 

loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size 

-> loss = F.cross_entropy(logits, b_labels.view(-1), reduction='mean')

안졍성 개선


### gradient clipping 사용

### weight decay 적용

### learning rate scheduler + warmup 적용