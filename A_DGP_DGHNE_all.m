function [features, P_G, P_D, Pt ] = A_DGP_DGHNE(features,DisIDset, AdjGfG,AdjGfD,AdjDfD ) 
    % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
	% features: embedding matrix; each row correspods to the embedding vector of a node; the first colomn records the node IDs.  
	% When features is empty, it will first pre-calculate the embedding matrix; this can decrease the time complexity.  
    % AdjGfG: associatins from (f) genes (G) to Genes (G)    
    % AdjGfD: associatins from Diseases (D) to Genes (G) GfD
    % AdjDfD  associatins from Diseases (D) to Disease (G) 
    % 
    % P0_G: column vector (set) initial probabilities in Gene network
    % P0_D: column vector (set) initial probabilities in Disease network
    % P0_G and P0_D must have the same # of columns.   
    % Ouput % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
    % P_G: stable probabilities in Gene network 
    % P_D: stable probabilities in Disease network  
    % Pt: stable probabilities in Gene+disease network 
    % % % % % % % % % % % % % % % % % % % % % %  
    % Reference: 
    % Xiang, et al, PrGeFNE: Predicting disease-related genes by fast network embedding, Methods, 2020,https://doi.org/10.1016/j.ymeth.2020.06.015. 
    % Zhang, et al, Billion-scale network embedding with iterative random projection. In 2018 IEEE International Conference on Data Mining (ICDM) (pp. 787-796). IEEE.

	knn_gene  = 100 ; 
	knn_dis   = 20  ;  
    % % % 
	restart  = 0.7 ; 
	pro_jump = 0.8 ;
	eta      = 0.5 ;   
    %  
    [N_gene, N_disease] = size( AdjGfD );
    if isempty( AdjDfD )
       AdjDfD = speye(N_disease);  
       warning('AdjDfD is empty.');
    end 
    % 
	if isempty( features )	
		Aunion = sparse( [AdjGfG,  AdjGfD; ...   
				          AdjGfD', AdjDfD;  ]  );      
		paras.d       = 128 ;
		paras.Ortho   = 1  ; 
		paras.seed    = 0; 
		worktype      = 'classification';    
		features      = getRandNEemb_in(Aunion,paras, [], [], worktype);    
		Aunion        = []; 
		if nargout<=1; 
			P_G = []; P_D=[]; Pt=[];  
			return; 
		end 
	end    
    % 
    ind_gene = [1: N_gene];
    ind_dis  = [N_gene+1: N_gene+N_disease];	
	%
    features_gene = features(ind_gene,2:end);  
    features_dis  = features(ind_dis,2:end);  
    features =[]; 
    %
    features_gene = features_gene./(max( abs( features_gene ), [] , 2) +eps ) ; 
    features_gene = features_gene./sqrt( sum( features_gene.^2 , 2) +eps ) ;   
    SimMatrix     = features_gene*features_gene';  
    %
    symmetrized   = true ; keepweight    = true ; 
	%AdjGfG(find(speye( size(AdjGfG) )) )= 1;  
    AdjGfG_rc     = sparse( getAdjKnnColumns_in( SimMatrix,  knn_gene , symmetrized, keepweight ) ) +AdjGfG ;  SimMatrix =[];     
    %
    features_dis  = features_dis./(max( abs( features_dis ), [] , 2) +eps ) ; 
    features_dis  = features_dis./sqrt( sum( features_dis.^2 , 2) +eps ) ;   
    SimMatrix     = features_dis*features_dis';      
	%AdjDfD(find(speye( size(AdjDfD) )) )= 1;  
    AdjDfD_rc     = sparse( getAdjKnnColumns_in( SimMatrix,  knn_dis , symmetrized, keepweight ) ) +AdjDfD;  SimMatrix =[];      
    %  
	%  	
    [ M_rc , IsNormalized ] = getNormalizedMatrix_Heter(AdjGfG_rc,AdjGfD,AdjDfD_rc, pro_jump,  'LaplacianNormalization', []) ; 
	% IsNormalized
	% 
	P0_G = AdjGfD(:,  DisIDset   );   
	P0_D = speye( size( AdjDfD) );  P0_D=P0_D(:,DisIDset); 	
	% 
 
    P0  = [ (1-eta)*P0_G; eta*P0_D];  
    Pt  = A_RWRplus(M_rc, restart, P0 , [],[], IsNormalized);   
    P_G = Pt(1:N_gene,:);
    P_D = Pt(N_gene+1:end,:); 
	% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
end
 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
function Adjknn = getAdjKnnColumns_in( SimMatrix,  k_neighbors_vec , symmetrized, keepweight )
% Input: 
% SimMatrix  similarity matrix
% k_neighbors  vector with number of neighbors of each node   
% symmetrized  1 or 0
% keepweight   keep similarity as weight of edges
% Output: Adjknn   matrix 
% Ju Xiang 
% 2019-5
    if isempty(SimMatrix) 
        % sort_dim = 2;
        SimMatrix =rand(10);   % for testing only 
        warning('test test test ');
    end 
    SZ =size( SimMatrix); 
    % 
    if isscalar(k_neighbors_vec)
        k_neighbors_vec = k_neighbors_vec(1)*ones( SZ(1), 1 ); 
    end
    if isempty(symmetrized) 
        symmetrized = true;
    end
    if isempty(keepweight) 
        keepweight = false;
    end
     
    if any( k_neighbors_vec>SZ(1)-1 )
        k_neighbors_vec(  k_neighbors_vec>SZ(1)   ) = SZ(1)-1;
        warning( ['There is k_neighbors:','>', num2str(SZ(1)-1),' the maximal number of neighbors'] );
    end
    
	% %    
	SimMatrix = SimMatrix + rand( SZ ).*( min(abs(SimMatrix(:)))/100000000 );  % 添加随机扰动  
    SimMatrix(   sub2ind( SZ, 1:SZ(1),1:SZ(1) )      ) = -inf;   %   
    % SimMatrix(   ( eye( SZ ) )==1       ) = -inf; 
    [~,II] = sort( SimMatrix ,2, 'descend' );  
    Adjknn = zeros( SZ );
    for ii=1: SZ(1)
        knn = II(ii,1: k_neighbors_vec(ii) ); 
        if keepweight
            Adjknn(ii,  knn ) = SimMatrix(ii,  knn );    
        else
            Adjknn(ii,  knn ) = 1; 
        end
    end 
    Adjknn(sub2ind( SZ, 1:SZ(1),1:SZ(1) )) = 0; 
    if symmetrized
        [i,j,v] = find( Adjknn ); 
        ind = sub2ind( SZ, i ,j );
        Adjknn = Adjknn' ; 
        Adjknn(ind) = v; 
        % %     Adjknn = Adjknn';
        % %     Adjknn(ind) = 
        % Adjknn = full
    end 
Adjknn=(Adjknn+Adjknn')/2;
end

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
function [emb]=getRandNEemb_in(A,paras, savefilename,TableNode, worktype)
    %  
    % % % % % % % % % % % % % % % % % % % % % % % % % % % 
    d       = paras.d;
    Ortho   = paras.Ortho;
    seed    = paras.seed;
    % % q       = paras.q;
    % % weights = paras.weights; 
    %
    N = length(A);
    % Common parameters
    % d = 128;
    % Ortho = 1;
    % seed = 0;
    switch worktype
        % worktype = 'classification';  'reconstruction';  
        case 'reconstruction'
            % embedding for adjacency matrix for reconstruction
            q = 2;
            weights = [1,0.1,0.001];
            U_list = RandNE_Projection(A,q,d,Ortho,seed);
            U = RandNE_Combine(U_list,weights);
            % prec = Precision_Np(A,sparse(N,N),U,U,1e6);
            % semilogx(1:1e6,prec);
        case 'classification'
            % % embedding for transition matrix for classification
            q = 3;
            weights = [1,1e2,1e4,1e5];
            A_tran = spdiags(1 ./ sum(A,2),0,N,N) * A;
            U_list = RandNE_Projection(A_tran,q,d,Ortho,seed);
            U = RandNE_Combine(U_list,weights);
            % % normalizing
            U = spdiags(1 ./ sqrt(sum(U .* U,2)),0,N,N) * U;
            % % Some Classification method, such as SVM in http://leitang.net/social_dimension.html
        otherwise
            error('No defintion')
    end
    [n_node, n_feature] = size( U ); 
    %  
	U1=(0:N-1)';
	emb=[U1 U];  
end
% % % % % % % % % % % % % % % % % % % % % % % % % 
function U_list = RandNE_Projection(A,q,d,Ortho,seed)
    % Inputs:
    %   A: sparse adjacency matrix
    %   q: order
    %   d: dimensionality
    %   Ortho: whether use orthogonal projection
    %   seed: random seed
    % Outputs:
    %   U_list: a list of R, A * R, A^2 * R ... A^q * R

    N = size(A,1);

    rng(seed);                               % set random seed
    U_list = cell(q + 1,1);                       % store each decomposed part
    U_list{1} = normrnd(0,1/sqrt(d),N,d);         % Gaussian random matrix
    if Ortho == 1                            % whether use orthogonal projection
        U_list{1} = GS(U_list{1});
    end
    for i = 2: (q + 1)                       % iterative random projection
        U_list{i} = A * U_list{i-1};
    end
end
 
% % % % % % % % % % % % % % % % % % % % % % % % % 
function P_ortho = GS(P)
    % Input:
    %   P: n x d random matrix
    % Output:
    %   P_ortho: each column orthogonal while maintaining length
    % Performing modified Gram?CSchmidt process

    [~,d] = size(P);
    temp_l = zeros(d,1);
    for i = 1:d
        temp_l(i) = sqrt( sum(P(:,i) .^2) );
    end
    for i = 1:d
        temp_row = P(:,i);
        for j = 1:i-1
            temp_j =  P(:,j);
            temp_product = temp_j' * temp_row  / temp_l(j)^2;
            temp_row = temp_row - temp_product * temp_j ; 
        end
        temp_row = temp_row * (temp_l(i) / sqrt(temp_row' * temp_row));
        P(:,i) = temp_row;
    end
    P_ortho = P;
end

% % % % % % % % % % % % % % % % % % % 
function U = RandNE_Combine(U_list,weights)
    % Inputs:
    %   U_list: a list of decomposed parts, generated by RandNE_Projection
    %   weights: a vector of weights for each order, w_0 ... w_q
    % Outputs:
    %   U: final embedding vector

    if size(U_list,1) < length(weights)
        error('Weights not consistent');
    end
    U = weights(1) * U_list{1};
    for i = 2:length(weights)
        U = U + (weights(i) * U_list{i});
    end

end
 
% % % % % % % % % % % 
% % % % % % % % % 
function WAdj = getNormalizedMatrix(Adj, NormalizationType, SetIsolatedNodeSelfLoop )
    if ~exist('Adj','var') 
        Adj =rand(5); dim=1;SetIsolatedNodeSelfLoop = true;  
        NormalizationType = 'col' ;
        % NormalizationType = 'laplacian normalization' ;
        istest = 1; 
        warning('Test Test Test Test Test Test Test ');
    end
% % % % % % % % % % % % % % % % 
% Adj  adjecent matrix  
% >= Matlab 2016    
% % % % % % % % % % % % % % %  
    if ischar(NormalizationType)
    %         NormalizationType =  (NormalizationType);
        switch  lower( NormalizationType )
            case lower( { 'column','col',  ...
                    'ProbabilityNormalizationColumn','ProbabilityNormalizationCol',...
                    'ProbabilityColumnNormalization','ProbabilityColNormalization',...
                    'NormalizationColumn','NormalizationCol' , ...
                    'ColumnNormalization','ColNormalization'   })
                NormalizationName = 'ProbabilityNormalization' ;  %  'Random Walk'  
                dim =1; 
            case lower('LaplacianNormalization')
                NormalizationName = NormalizationType;  
            case lower({'none', 'None', 'NONE'})
                NormalizationName = 'None'; 
                WAdj = Adj; 
                return; 
            otherwise
                error(['There is no type of normalization: ',char( string(NormalizationType) )] );
        end 
    else; error('There is no defintion of NormalizationType')
    end 
    % NormalizationName = lower( NormalizationName ); 
    switch lower( NormalizationName )
        case lower( 'ProbabilityNormalization' )
            degrees = sum(Adj,dim);
            if any( degrees~=1)
                WAdj = Adj./ ( degrees+eps  );           
                % % WAdj = Adj./ repmat( degrees +eps,[size(Adj,1),1]); 
            else
                WAdj = Adj; 
            end
            % 
            if SetIsolatedNodeSelfLoop  && size(Adj,1)==size(Adj,2) 
                ii = find( ~degrees ); 
                idx = sub2ind( size(Adj), ii,ii ); 
                WAdj(idx) = 1;  % set to be 1 for isolated nodes, 
            end
            
        case lower( 'LaplacianNormalization')
            deg_rowvec = ( sum(Adj,1) ).^0.5;  
            deg_colvec = ( sum(Adj,2) ).^0.5;   
            WAdj = (Adj./(deg_colvec+eps))./(deg_rowvec+eps) ;    
            % 
            if SetIsolatedNodeSelfLoop && size(Adj,1)==size(Adj,2)
                ii = find( ~sum(Adj,2) ) ;  
                WAdj( sub2ind( size(Adj), ii,ii ) ) = 1;  % set to be 1 for isolated nodes, 
            end 
			
        case lower( {'None','none'} )
            WAdj = Adj;   
        otherwise
            error(['NormalizationName is wrong: ',char(string(NormalizationName) )   ]);
    end
% % % % % % 
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ combMatrix , IsNormalized ] = getNormalizedMatrix_Heter(AdjGfG,AdjGfD,AdjDfD, pro_jump,  NormalizationType, isdebug) 
% % % % % % % % % % % % % % % % % % % % 
% Input % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% AdjGfG: associatins from (f) genes (G) to Genes (G)  
% AdjGfD: associatins from Diseases (D) to Genes (G) GfD
% AdjDfD  associatins from Diseases (D) to Disease (G)  
% pro_jump: jumping Probability from first layer to second layer or weighting the effect of second layer on the first layer.   
% NormalizationType = 'LaplacianNormalization'; %%  for label propagation, prince and more....    
% NormalizationType = 'Weight'; %% Weighting  ....    
% NormalizationType = 'None'; %%  without normalization ....    
% Ouput % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% combMatrix is matrix after normalization.   
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% By Xiang  
% 2019-8-2 
    % global   Global_Var_RWRH 
    if ~exist('pro_jump','var') || isempty (pro_jump)
        pro_jump = 0.5; 
    elseif pro_jump>1 || pro_jump <0
        error('pro_jump is wrong. it should be between 0 and 1');
    end    
   if ~exist('isdebug','var') || isempty (isdebug)
        isdebug = false;  
    end   
    %  
    [N_gene, N_disease] = size( AdjGfD );
    if isempty( AdjDfD )
       AdjDfD = speye(N_disease);  
       warning('AdjDfD is empty.');
    end 
    %
    if ~exist('NormalizationType','var') || isempty(NormalizationType)
        NormalizationType = 'None'; 
    end
    %  
    IsNormalized = true; 
    switch lower( NormalizationType )
        case lower( {'None'} )
            combMatrix = [ AdjGfG, AdjGfD; AdjGfD', AdjDfD    ] ;
            IsNormalized = false;
            
        case lower( {'Weight'} )  
            combMatrix = [ (1-pro_jump).*AdjGfG, pro_jump.*AdjGfD; pro_jump.*AdjGfD', (1-pro_jump).*AdjDfD    ] ;
            IsNormalized = false;
 
        case lower( {'LaplacianNormalization'} ) 
            % WAdj = getNormalizedMatrix(Adj, NormalizationType, SetIsolatedNodeSelfLoop )
            M_GfG = getNormalizedMatrix(AdjGfG   , NormalizationType, false );   
            M_DfD = getNormalizedMatrix(AdjDfD   , NormalizationType, false ); 
            M_GfD = getNormalizedMatrix(AdjGfD   , NormalizationType, false );  % probabilities from disease space to gene space 
            M_DfG = getNormalizedMatrix(AdjGfD'  , NormalizationType, false );  % probabilities from gene space to disease space
            %
            combMatrix = [ (1-pro_jump).*M_GfG, pro_jump.*M_GfD; pro_jump.*M_DfG, (1-pro_jump).*M_DfD    ] ;            
            
        otherwise
            error('No definition.');
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function  [Pt, WAdj ]= A_RWRplus(Adj, r_restart, P0, N_max_iter, Eps_min_change, IsNormalized,  NormalizationType)
% Ouput
% Pt  
% WAdj   normalized Adj 
% % % % % % % % % % % % % %   
    if ~exist('N_max_iter','var') || isempty(N_max_iter) || (isnumeric( N_max_iter) && N_max_iter<=1 ) 
        N_max_iter =100; 
    elseif ~isnumeric( N_max_iter)  
        error('N_max_iter should be isnumeric!!!!!') ;
    end
    %
    if ~exist('Eps_min_change','var') || isempty(Eps_min_change) 
        Eps_min_change =10^-6; 
    elseif isnumeric( Eps_min_change) && Eps_min_change>=1 
        warning('The Eps_min_change is nomenaning. Reset Eps_min_change to be 10^-6.'); 
        Eps_min_change =10^-6;  
    elseif ~isnumeric( Eps_min_change)  
        error('Eps_min_change should be isnumeric!!!!!') ;
    end
    
    if ~exist('IsNormalized','var') || isempty(IsNormalized) 
        IsNormalized = false;  % Adj has been normalized for fast run.   
    end
    
    if ~exist('NormalizationType','var') || isempty(NormalizationType) 
        NormalizationType = 'ProbabilityNormalizationColumn'; %%for 'Random Walk' RWR, RWRH  RWRM  RWRMH and more   
    end        
	% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %    
    P0  = full(P0); 
    %   
    if IsNormalized 
        WAdj = Adj; 
    else 
        switch NormalizationType
            case {'ProbabilityNormalizationColumn','ProbabilityNormalizationCol','col','column'} 
                WAdj = getNormalizedMatrix(Adj, 'col', true ); 
                
            case 'LaplacianNormalization'   
                WAdj = getNormalizedMatrix(Adj, 'LaplacianNormalization', true );   
                  
            otherwise
                error(['NormalizationType is wrong: ',char( string(NormalizationType) )]); 
        end        
    end   
    % % % % % % % % % % % % % % % % % % %    
    % % Solver_IterationPropagation
    % % It can be used directly when IsNormalized is TRUE.  
    Pt = P0;
    for T = 1: N_max_iter
        Pt1 = (1-r_restart)*WAdj*Pt + r_restart*P0;
        if all( sum( abs( Pt1-Pt )) < Eps_min_change )
            break;
        end
        Pt = Pt1;
    end
    Pt = full(Pt); 
end
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         

