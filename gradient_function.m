function [grad_res] =  gradient_function(theta, X, y, len,lambda)
    h_theta = sigmoid(X*theta);
    J=sum(-y.*log(h_theta)-(1-y).*log(1-h_theta))/len+sum(theta.^2)*lambda/2/len;
    for i = 1:44
        grad_res(i,:)=(sum((h_theta-y).*X(:,i),1)./len)'+[theta(i,:).*lambda./len];
    end
    
end

